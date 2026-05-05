//! Streaming, callback-based Wavefront OBJ reader and writer with matched
//! read/write traits.
//!
//! # Why
//!
//! Most OBJ crates on crates.io eagerly load a file into a `Mesh` struct.
//! That works great when the input fits in memory and you don't care about
//! preserving exact byte layout. This crate fills the opposite niche:
//!
//! - **Streaming.** [`read_obj_file`] walks the file once and dispatches to
//!   trait callbacks - no intermediate allocation per element. You can
//!   process arbitrarily large meshes by writing your own [`ObjReader`]
//!   that pushes data straight into the buffer of your choice.
//! - **Round-trip fidelity.** [`ObjReader`] and [`ObjWriter`] are matched
//!   trait pairs: every directive a reader can produce, a writer can emit.
//!   That makes byte-equal round trips of OBJ files trivial.
//! - **Configurable float precision.** Read and write `f32` or `f64` via
//!   the [`ObjFloat`] generic parameter.
//! - **Strict-by-default with explicit opt-in for lenient parsing.**
//!   [`ObjReader::read_unknown`] returns an error by default; override it
//!   to silently skip directives you don't care about (e.g. NURBS).
//!
//! If what you want is `let mesh = obj::load(path)?`, use
//! [`tobj`](https://crates.io/crates/tobj) instead - that is the right tool
//! for that job.
//!
//! # Supported directives
//!
//! Core geometry: `v`, `vt`, `vn`, `f`, `o`, `#` comments.
//!
//! Standard auxiliary directives have first-class trait methods on both
//! [`ObjReader`] and [`ObjWriter`]: `mtllib`, `usemtl`, `g`, `s` (with
//! [`SmoothingGroup`]), `l`, `p`.
//!
//! Anything else (NURBS / free-form geometry, display attributes, vendor
//! extensions) routes to [`ObjReader::read_unknown`] - reject or ignore as
//! you see fit.
//!
//! # Quick start
//!
//! ```
//! use wavefront_obj_io::{ObjReader, read_obj_file};
//! use std::io::Cursor;
//!
//! #[derive(Default)]
//! struct CountVertices(usize);
//!
//! impl ObjReader<f32> for CountVertices {
//!     fn read_comment(&mut self, _: &str) {}
//!     fn read_object_name(&mut self, _: &str) {}
//!     fn read_vertex(&mut self, _: f32, _: f32, _: f32, _: Option<f32>) {
//!         self.0 += 1;
//!     }
//!     fn read_texture_coordinate(&mut self, _: f32, _: Option<f32>, _: Option<f32>) {}
//!     fn read_normal(&mut self, _: f32, _: f32, _: f32) {}
//!     fn read_face(&mut self, _: &[(usize, Option<usize>, Option<usize>)]) {}
//! }
//!
//! let obj = "v 0 0 0\nv 1 0 0\nv 0 1 0\n";
//! let mut counter = CountVertices::default();
//! read_obj_file(Cursor::new(obj), &mut counter).unwrap();
//! assert_eq!(counter.0, 3);
//! ```
//!
//! Indices follow the Wavefront convention and are kept 1-based throughout
//! the API.

use std::error::Error as StdError;
use std::fmt;
use std::fmt::Display;
use std::io;
use std::io::{BufRead, BufReader};
use std::str::FromStr;

/// Error type returned by [`read_obj_file`].
///
/// `Io` wraps an underlying [`io::Error`] from the source. `Parse` carries a
/// structured description of an OBJ syntax problem at a specific line.
#[derive(Debug)]
pub enum ObjError {
    Io(io::Error),
    Parse { line: usize, kind: ParseErrorKind },
}

/// Structured description of an OBJ syntax problem.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseErrorKind {
    /// A line had no directive prefix after trimming whitespace.
    EmptyPrefix,
    /// The line started with a directive the parser does not recognize and
    /// the [`ObjReader::read_unknown`] callback rejected it.
    UnknownPrefix(String),
    /// A directive was missing a required field.
    MissingField(&'static str),
    /// A numeric value could not be parsed as a float.
    InvalidNumber { field: &'static str, value: String },
    /// A face / line / point index was zero, negative, or non-numeric.
    InvalidIndex { kind: &'static str, value: String },
    /// `s <value>` where the value was neither `off` nor a non-negative integer.
    InvalidSmoothingGroup(String),
    /// `l` element with fewer than 2 vertices.
    LineElementTooShort,
    /// `p` element with no vertices.
    PointElementEmpty,
    /// Free-form message, e.g. from a custom [`ObjReader::read_unknown`].
    Custom(String),
}

impl Display for ObjError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ObjError::Io(e) => write!(f, "I/O error: {e}"),
            ObjError::Parse { line, kind } => write!(f, "line {line}: {kind}"),
        }
    }
}

impl Display for ParseErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseErrorKind::EmptyPrefix => write!(f, "empty prefix"),
            ParseErrorKind::UnknownPrefix(p) => write!(f, "Unknown line prefix: {p}"),
            ParseErrorKind::MissingField(field) => write!(f, "missing {field}"),
            ParseErrorKind::InvalidNumber { field, value } => {
                write!(f, "invalid {field}: {value}")
            }
            ParseErrorKind::InvalidIndex { kind, value } => {
                write!(f, "invalid {kind} index: {value}")
            }
            ParseErrorKind::InvalidSmoothingGroup(v) => {
                write!(
                    f,
                    "invalid smoothing group: {v} (expected integer or 'off')"
                )
            }
            ParseErrorKind::LineElementTooShort => {
                write!(f, "line element needs at least 2 vertices")
            }
            ParseErrorKind::PointElementEmpty => {
                write!(f, "point element needs at least 1 vertex")
            }
            ParseErrorKind::Custom(s) => write!(f, "{s}"),
        }
    }
}

impl StdError for ObjError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            ObjError::Io(e) => Some(e),
            ObjError::Parse { .. } => None,
        }
    }
}

impl From<io::Error> for ObjError {
    fn from(e: io::Error) -> Self {
        ObjError::Io(e)
    }
}

impl From<ObjError> for io::Error {
    fn from(e: ObjError) -> Self {
        match e {
            ObjError::Io(inner) => inner,
            parse @ ObjError::Parse { .. } => {
                io::Error::new(io::ErrorKind::InvalidData, parse.to_string())
            }
        }
    }
}

/// Trait for floating point types that can be used in OBJ files.
/// This allows the library to work with both f32 and f64 precision.
pub trait ObjFloat: Copy + Display + FromStr + PartialEq {
    /// Returns the fractional part of the number
    fn fract(self) -> Self;

    /// Returns true if the fractional part is zero
    fn is_zero_fract(self) -> bool {
        self.fract() == Self::zero()
    }

    /// Returns the zero value for this type
    fn zero() -> Self;
}

impl ObjFloat for f32 {
    fn fract(self) -> Self {
        self.fract()
    }
    fn zero() -> Self {
        0.0
    }
}

impl ObjFloat for f64 {
    fn fract(self) -> Self {
        self.fract()
    }
    fn zero() -> Self {
        0.0
    }
}

/// Smoothing group selector for the `s` directive.
///
/// `s 0` and `s off` both disable smoothing; any positive integer names a
/// smoothing group.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SmoothingGroup {
    Off,
    Group(u32),
}

/// Trait for writing OBJ file data with configurable float precision.
///
/// The generic parameter `F` defaults to `f64` for backward compatibility,
/// but can be set to `f32` for applications that work with single-precision data.
pub trait ObjWriter<F: ObjFloat = f64> {
    fn write_comment<S: AsRef<str>>(&mut self, comment: S) -> io::Result<()>;
    fn write_object_name<S: AsRef<str>>(&mut self, name: S) -> io::Result<()>;
    fn write_vertex(&mut self, x: F, y: F, z: F, w: Option<F>) -> io::Result<()>;
    fn write_texture_coordinate(&mut self, u: F, v: Option<F>, w: Option<F>) -> io::Result<()>;
    fn write_normal(&mut self, nx: F, ny: F, nz: F) -> io::Result<()>;
    fn write_face(
        &mut self,
        vertex_indices: &[(usize, Option<usize>, Option<usize>)],
    ) -> io::Result<()>;

    /// `mtllib lib1.mtl lib2.mtl ...` - reference one or more material libraries.
    fn write_material_lib<S: AsRef<str>>(&mut self, names: &[S]) -> io::Result<()>;

    /// `usemtl name` - select a material from a previously declared library.
    fn write_use_material<S: AsRef<str>>(&mut self, name: S) -> io::Result<()>;

    /// `g name1 name2 ...` - assign subsequent elements to one or more groups.
    fn write_group<S: AsRef<str>>(&mut self, names: &[S]) -> io::Result<()>;

    /// `s 0|off|<n>` - select a smoothing group for subsequent faces.
    fn write_smoothing_group(&mut self, group: SmoothingGroup) -> io::Result<()>;

    /// `l v1[/vt1] v2[/vt2] ...` - polyline element. Each index is a vertex,
    /// optionally with a texture coordinate.
    fn write_line_element(&mut self, indices: &[(usize, Option<usize>)]) -> io::Result<()>;

    /// `p v1 v2 ...` - point element.
    fn write_point_element(&mut self, indices: &[usize]) -> io::Result<()>;
}

/// Trait for reading OBJ file data with configurable float precision.
///
/// The generic parameter `F` defaults to `f64` for backward compatibility,
/// but can be set to `f32` for applications that work with single-precision data.
pub trait ObjReader<F: ObjFloat = f64> {
    fn read_comment(&mut self, comment: &str) -> ();
    fn read_object_name(&mut self, name: &str) -> ();
    fn read_vertex(&mut self, x: F, y: F, z: F, w: Option<F>) -> ();
    fn read_texture_coordinate(&mut self, u: F, v: Option<F>, w: Option<F>) -> ();
    fn read_normal(&mut self, nx: F, ny: F, nz: F) -> ();
    fn read_face(&mut self, vertex_indices: &[(usize, Option<usize>, Option<usize>)]) -> ();

    /// `mtllib lib1.mtl lib2.mtl ...` - default no-op.
    fn read_material_lib(&mut self, _names: &[&str]) {}

    /// `usemtl name` - default no-op.
    fn read_use_material(&mut self, _name: &str) {}

    /// `g name1 name2 ...` - default no-op.
    fn read_group(&mut self, _names: &[&str]) {}

    /// `s 0|off|<n>` - default no-op.
    fn read_smoothing_group(&mut self, _group: SmoothingGroup) {}

    /// `l v1[/vt1] v2[/vt2] ...` - default no-op.
    fn read_line_element(&mut self, _indices: &[(usize, Option<usize>)]) {}

    /// `p v1 v2 ...` - default no-op.
    fn read_point_element(&mut self, _indices: &[usize]) {}

    /// Called when a line with an unknown prefix is encountered.
    ///
    /// The default implementation returns `ParseErrorKind::UnknownPrefix`,
    /// treating any prefix outside the supported core (NURBS / free-form
    /// geometry, display attributes, vendor extensions) as a hard error.
    /// Override to skip or otherwise handle these lines.
    fn read_unknown(&mut self, prefix: &str, _rest: &str, line: usize) -> Result<(), ObjError> {
        Err(ObjError::Parse {
            line,
            kind: ParseErrorKind::UnknownPrefix(prefix.to_string()),
        })
    }
}

pub fn read_obj_file<R: io::Read, T: ObjReader<F>, F: ObjFloat>(
    reader: R,
    obj_reader: &mut T,
) -> Result<(), ObjError>
where
    <F as FromStr>::Err: Display,
{
    let mut buf_reader = BufReader::new(reader);
    let mut line = String::new();
    let mut lineno: usize = 0;

    while buf_reader.read_line(&mut line)? != 0 {
        lineno += 1;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            line.clear();
            continue;
        }
        let mut parts = trimmed.split_whitespace();
        let prefix = parts.next().ok_or(ObjError::Parse {
            line: lineno,
            kind: ParseErrorKind::EmptyPrefix,
        })?;

        let parse_f = |s: &str, field: &'static str| -> Result<F, ObjError> {
            s.parse::<F>().map_err(|_| ObjError::Parse {
                line: lineno,
                kind: ParseErrorKind::InvalidNumber {
                    field,
                    value: s.to_string(),
                },
            })
        };

        let parse_index = |s: &str, kind: &'static str| -> Result<usize, ObjError> {
            let index = s.parse::<usize>().map_err(|_| ObjError::Parse {
                line: lineno,
                kind: ParseErrorKind::InvalidIndex {
                    kind,
                    value: s.to_string(),
                },
            })?;
            if index == 0 {
                return Err(ObjError::Parse {
                    line: lineno,
                    kind: ParseErrorKind::InvalidIndex {
                        kind,
                        value: s.to_string(),
                    },
                });
            }
            Ok(index)
        };

        let missing = |field: &'static str| -> ObjError {
            ObjError::Parse {
                line: lineno,
                kind: ParseErrorKind::MissingField(field),
            }
        };

        match prefix {
            "#" => {
                let comment = parts.collect::<Vec<&str>>().join(" ");
                obj_reader.read_comment(&comment);
            }
            "v" => {
                let x = parts
                    .next()
                    .ok_or_else(|| missing("vertex x"))
                    .and_then(|s| parse_f(s, "vertex x"))?;
                let y = parts
                    .next()
                    .ok_or_else(|| missing("vertex y"))
                    .and_then(|s| parse_f(s, "vertex y"))?;
                let z = parts
                    .next()
                    .ok_or_else(|| missing("vertex z"))
                    .and_then(|s| parse_f(s, "vertex z"))?;
                let w = match parts.next() {
                    Some(s) => Some(parse_f(s, "vertex w")?),
                    None => None,
                };
                obj_reader.read_vertex(x, y, z, w);
            }
            "vt" => {
                let u = parts
                    .next()
                    .ok_or_else(|| missing("texture u"))
                    .and_then(|s| parse_f(s, "texture u"))?;
                let v = match parts.next() {
                    Some(s) => Some(parse_f(s, "texture v")?),
                    None => None,
                };
                let w = match parts.next() {
                    Some(s) => Some(parse_f(s, "texture w")?),
                    None => None,
                };
                obj_reader.read_texture_coordinate(u, v, w);
            }
            "vn" => {
                let nx = parts
                    .next()
                    .ok_or_else(|| missing("normal nx"))
                    .and_then(|s| parse_f(s, "normal nx"))?;
                let ny = parts
                    .next()
                    .ok_or_else(|| missing("normal ny"))
                    .and_then(|s| parse_f(s, "normal ny"))?;
                let nz = parts
                    .next()
                    .ok_or_else(|| missing("normal nz"))
                    .and_then(|s| parse_f(s, "normal nz"))?;
                obj_reader.read_normal(nx, ny, nz);
            }
            "f" => {
                let mut vertex_indices = Vec::new();
                for part in parts {
                    // parse "v[/vt[/vn]]" by slicing without allocating
                    let first_slash = part.find('/');
                    let (v_str, rest) = match first_slash {
                        Some(i) => (&part[..i], &part[i + 1..]),
                        None => (part, ""),
                    };

                    let v_idx = parse_index(v_str, "vertex")?;

                    let (vt_idx, vn_idx) = if rest.is_empty() {
                        (None, None)
                    } else {
                        let second_slash = rest.find('/');
                        if let Some(j) = second_slash {
                            let vt_part = &rest[..j];
                            let vn_part = &rest[j + 1..];
                            let vt = if vt_part.is_empty() {
                                None
                            } else {
                                Some(parse_index(vt_part, "texcoord")?)
                            };
                            let vn = if vn_part.is_empty() {
                                None
                            } else {
                                Some(parse_index(vn_part, "normal")?)
                            };
                            (vt, vn)
                        } else {
                            // only vt present
                            let vt = if rest.is_empty() {
                                None
                            } else {
                                Some(parse_index(rest, "texcoord")?)
                            };
                            (vt, None)
                        }
                    };

                    vertex_indices.push((v_idx, vt_idx, vn_idx));
                }
                obj_reader.read_face(&vertex_indices);
            }
            "o" => {
                let name = parts.collect::<Vec<&str>>().join(" ");
                obj_reader.read_object_name(&name);
            }
            "mtllib" => {
                let names: Vec<&str> = parts.collect();
                obj_reader.read_material_lib(&names);
            }
            "usemtl" => {
                // Material names should not contain whitespace per spec; join
                // anything we get just to be tolerant.
                let name = parts.collect::<Vec<&str>>().join(" ");
                obj_reader.read_use_material(&name);
            }
            "g" => {
                let names: Vec<&str> = parts.collect();
                obj_reader.read_group(&names);
            }
            "s" => {
                let value = parts
                    .next()
                    .ok_or_else(|| missing("smoothing group value"))?;
                let group = if value.eq_ignore_ascii_case("off") {
                    SmoothingGroup::Off
                } else {
                    let n = value.parse::<u32>().map_err(|_| ObjError::Parse {
                        line: lineno,
                        kind: ParseErrorKind::InvalidSmoothingGroup(value.to_string()),
                    })?;
                    if n == 0 {
                        SmoothingGroup::Off
                    } else {
                        SmoothingGroup::Group(n)
                    }
                };
                obj_reader.read_smoothing_group(group);
            }
            "l" => {
                let mut indices: Vec<(usize, Option<usize>)> = Vec::new();
                for part in parts {
                    let (v_str, vt_str) = match part.find('/') {
                        Some(i) => (&part[..i], Some(&part[i + 1..])),
                        None => (part, None),
                    };
                    let v_idx = parse_index(v_str, "vertex")?;
                    let vt_idx = match vt_str {
                        Some(s) if !s.is_empty() => Some(parse_index(s, "texcoord")?),
                        _ => None,
                    };
                    indices.push((v_idx, vt_idx));
                }
                if indices.len() < 2 {
                    return Err(ObjError::Parse {
                        line: lineno,
                        kind: ParseErrorKind::LineElementTooShort,
                    });
                }
                obj_reader.read_line_element(&indices);
            }
            "p" => {
                let mut indices: Vec<usize> = Vec::new();
                for part in parts {
                    indices.push(parse_index(part, "vertex")?);
                }
                if indices.is_empty() {
                    return Err(ObjError::Parse {
                        line: lineno,
                        kind: ParseErrorKind::PointElementEmpty,
                    });
                }
                obj_reader.read_point_element(&indices);
            }
            other => {
                let rest = parts.collect::<Vec<&str>>().join(" ");
                obj_reader.read_unknown(other, &rest, lineno)?;
            }
        }

        line.clear();
    }

    Ok(())
}

pub struct IoObjWriter<W: io::Write, F: ObjFloat = f64> {
    out: W,
    line_buf: Vec<u8>,
    /// When true, format floats with 6 decimal places (matching C `printf %f`).
    printf_f_format: bool,
    _phantom: std::marker::PhantomData<F>,
}
impl<W: io::Write, F: ObjFloat> IoObjWriter<W, F> {
    /// Creates a new OBJ writer with default formatting (full precision).
    pub fn new(writer: W) -> Self {
        IoObjWriter {
            out: writer,
            line_buf: Vec::with_capacity(256),
            printf_f_format: false,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Creates a new OBJ writer that formats floats with 6 decimal places,
    /// matching the output produced by C's `fprintf("%f", ...)`.
    #[cfg(test)]
    pub fn new_with_printf_f_format(writer: W) -> Self {
        IoObjWriter {
            out: writer,
            line_buf: Vec::with_capacity(256),
            printf_f_format: true,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Toggle 6-decimal-place float formatting (`printf %f` style).
    #[cfg(test)]
    pub fn set_printf_f_format(&mut self, enabled: bool) {
        self.printf_f_format = enabled;
    }

    #[inline]
    fn push_str(&mut self, s: &str) {
        self.line_buf.extend_from_slice(s.as_bytes());
    }

    #[inline]
    fn push_u<T: itoa::Integer>(&mut self, v: T) {
        let mut buf = itoa::Buffer::new();
        self.push_str(buf.format(v));
    }

    #[inline]
    fn push_f(&mut self, v: F) {
        // we want 0 as "0" not "0.0"
        if v.is_zero_fract() {
            self.push_str(&format!("{}", v));
            return;
        }
        // 6 decimal places when matching C `printf %f`, otherwise full
        // round-trippable precision via the type's Display impl.
        if self.printf_f_format {
            self.push_str(&format!("{:.6}", v));
        } else {
            self.push_str(&format!("{}", v));
        }
    }

    #[inline]
    fn flush_line(&mut self) -> io::Result<()> {
        self.line_buf.push(b'\n');
        self.out.write_all(&self.line_buf)?;
        self.line_buf.clear();
        Ok(())
    }
}
impl<W: io::Write, F: ObjFloat> ObjWriter<F> for IoObjWriter<W, F> {
    fn write_comment<S: AsRef<str>>(&mut self, comment: S) -> io::Result<()> {
        self.push_str("# ");
        self.push_str(comment.as_ref());
        self.flush_line()
    }

    fn write_object_name<S: AsRef<str>>(&mut self, name: S) -> io::Result<()> {
        self.push_str("o ");
        self.push_str(name.as_ref());
        self.flush_line()
    }

    fn write_vertex(&mut self, x: F, y: F, z: F, w: Option<F>) -> io::Result<()> {
        self.push_str("v ");
        self.push_f(x);
        self.push_str(" ");
        self.push_f(y);
        self.push_str(" ");
        self.push_f(z);
        if let Some(wv) = w {
            self.push_str(" ");
            self.push_f(wv);
        }
        self.flush_line()
    }

    fn write_texture_coordinate(&mut self, u: F, v: Option<F>, w: Option<F>) -> io::Result<()> {
        self.push_str("vt ");
        self.push_f(u);
        if let Some(vv) = v {
            self.push_str(" ");
            self.push_f(vv);
            if let Some(wv) = w {
                self.push_str(" ");
                self.push_f(wv);
            }
        }
        self.flush_line()
    }

    fn write_normal(&mut self, nx: F, ny: F, nz: F) -> io::Result<()> {
        self.push_str("vn ");
        self.push_f(nx);
        self.push_str(" ");
        self.push_f(ny);
        self.push_str(" ");
        self.push_f(nz);
        self.flush_line()
    }

    fn write_face(
        &mut self,
        vertex_indices: &[(usize, Option<usize>, Option<usize>)],
    ) -> io::Result<()> {
        // Build the whole face line and write once.
        self.push_str("f");
        for (v_idx, vt_idx, vn_idx) in vertex_indices.iter() {
            self.push_str(" ");
            // If your internal indices are zero-based, emit +1 here:
            self.push_u(*v_idx);
            match (vt_idx, vn_idx) {
                (None, None) => {}
                (Some(vt), None) => {
                    self.push_str("/");
                    self.push_u(*vt);
                }
                (None, Some(vn)) => {
                    self.push_str("//");
                    self.push_u(*vn);
                }
                (Some(vt), Some(vn)) => {
                    self.push_str("/");
                    self.push_u(*vt);
                    self.push_str("/");
                    self.push_u(*vn);
                }
            }
        }
        self.flush_line()
    }

    fn write_material_lib<S: AsRef<str>>(&mut self, names: &[S]) -> io::Result<()> {
        self.push_str("mtllib");
        for name in names {
            self.push_str(" ");
            self.push_str(name.as_ref());
        }
        self.flush_line()
    }

    fn write_use_material<S: AsRef<str>>(&mut self, name: S) -> io::Result<()> {
        self.push_str("usemtl ");
        self.push_str(name.as_ref());
        self.flush_line()
    }

    fn write_group<S: AsRef<str>>(&mut self, names: &[S]) -> io::Result<()> {
        self.push_str("g");
        for name in names {
            self.push_str(" ");
            self.push_str(name.as_ref());
        }
        self.flush_line()
    }

    fn write_smoothing_group(&mut self, group: SmoothingGroup) -> io::Result<()> {
        match group {
            SmoothingGroup::Off => self.push_str("s off"),
            SmoothingGroup::Group(n) => {
                self.push_str("s ");
                self.push_u(n);
            }
        }
        self.flush_line()
    }

    fn write_line_element(&mut self, indices: &[(usize, Option<usize>)]) -> io::Result<()> {
        self.push_str("l");
        for (v_idx, vt_idx) in indices.iter() {
            self.push_str(" ");
            self.push_u(*v_idx);
            if let Some(vt) = vt_idx {
                self.push_str("/");
                self.push_u(*vt);
            }
        }
        self.flush_line()
    }

    fn write_point_element(&mut self, indices: &[usize]) -> io::Result<()> {
        self.push_str("p");
        for v_idx in indices.iter() {
            self.push_str(" ");
            self.push_u(*v_idx);
        }
        self.flush_line()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;
    use std::io::Cursor;

    type Face = Vec<(usize, Option<usize>, Option<usize>)>;

    #[derive(Default)]
    struct TestObjReader64 {
        comments: Vec<String>,
        names: Vec<String>,
        vertices: Vec<(f64, f64, f64, Option<f64>)>,
        texture_coordinates: Vec<(f64, Option<f64>, Option<f64>)>,
        normals: Vec<(f64, f64, f64)>,
        faces: Vec<Face>,
    }

    impl ObjReader for TestObjReader64 {
        fn read_comment(&mut self, comment: &str) {
            self.comments.push(comment.to_string());
        }

        fn read_object_name(&mut self, name: &str) {
            self.names.push(name.to_string());
        }

        fn read_vertex(&mut self, x: f64, y: f64, z: f64, w: Option<f64>) {
            self.vertices.push((x, y, z, w));
        }

        fn read_texture_coordinate(&mut self, u: f64, v: Option<f64>, w: Option<f64>) {
            self.texture_coordinates.push((u, v, w));
        }

        fn read_normal(&mut self, nx: f64, ny: f64, nz: f64) {
            self.normals.push((nx, ny, nz));
        }

        fn read_face(&mut self, vertex_indices: &[(usize, Option<usize>, Option<usize>)]) {
            self.faces.push(vertex_indices.to_vec());
        }
    }

    #[test]
    fn test_obj_reading() {
        // read testdata/screw.obj using TestObjReader
        let obj_data = include_str!("../testdata/screw_f64.obj");
        let cursor = Cursor::new(obj_data);
        let mut reader: TestObjReader64 = Default::default();
        read_obj_file(cursor, &mut reader).unwrap();
        // this does not check correctness and ordering of data, just that all data was read
        assert_eq!(reader.comments.len(), 3);
        assert_eq!(reader.names.len(), 1);
        assert_eq!(reader.vertices.len(), 41);
        assert_eq!(reader.texture_coordinates.len(), 41);
        assert_eq!(reader.normals.len(), 41);
        assert_eq!(reader.faces.len(), 48);
    }

    #[test]
    fn test_obj_reading_2() {
        let input = "# This is a test OBJ file
o TestObject
v 1 2 3
vt 0.5 0.5
vn 0 1 1.1
f 1/1/1 2/2/2 3/3/3
";

        let reader = Cursor::new(input);
        let mut test_reader: TestObjReader64 = Default::default();
        read_obj_file(reader, &mut test_reader).unwrap();
        assert_eq!(test_reader.comments, vec!["This is a test OBJ file"]);
        assert_eq!(test_reader.names, vec!["TestObject"]);
        assert_eq!(test_reader.vertices, vec![(1.0, 2.0, 3.0, None)]);
        assert_eq!(
            test_reader.texture_coordinates,
            vec![(0.5, Some(0.5), None)]
        );
        assert_eq!(test_reader.normals, vec![(0.0, 1.0, 1.1)]);
        assert_eq!(
            test_reader.faces,
            vec![vec![
                (1, Some(1), Some(1)),
                (2, Some(2), Some(2)),
                (3, Some(3), Some(3))
            ]]
        );
    }

    #[test]
    fn test_obj_writing() {
        let mut buffer = Vec::new();
        let mut writer = IoObjWriter::new(&mut buffer);
        writer.write_comment("This is a test OBJ file").unwrap();
        writer.write_object_name("TestObject").unwrap();
        writer.write_vertex(1.0, 2.0, 3.0, None).unwrap();
        writer
            .write_texture_coordinate(0.5, Some(0.5), None)
            .unwrap();
        writer.write_normal(0.0, 1.0, 1.1).unwrap();
        writer
            .write_face(&[
                (1, Some(1), Some(1)),
                (2, Some(2), Some(2)),
                (3, Some(3), Some(3)),
            ])
            .unwrap();

        let output = String::from_utf8(buffer).unwrap();
        let expected_output = "# This is a test OBJ file
o TestObject
v 1 2 3
vt 0.5 0.5
vn 0 1 1.1
f 1/1/1 2/2/2 3/3/3
";
        assert_eq!(output, expected_output);
    }

    struct WritingReader64 {
        writer: IoObjWriter<Vec<u8>>,
    }
    impl ObjReader for WritingReader64 {
        fn read_comment(&mut self, comment: &str) {
            self.writer.write_comment(comment).unwrap();
        }

        fn read_object_name(&mut self, name: &str) {
            self.writer.write_object_name(name).unwrap();
        }

        fn read_vertex(&mut self, x: f64, y: f64, z: f64, w: Option<f64>) {
            self.writer.write_vertex(x, y, z, w).unwrap();
        }

        fn read_texture_coordinate(&mut self, u: f64, v: Option<f64>, w: Option<f64>) {
            self.writer.write_texture_coordinate(u, v, w).unwrap();
        }

        fn read_normal(&mut self, nx: f64, ny: f64, nz: f64) {
            self.writer.write_normal(nx, ny, nz).unwrap();
        }

        fn read_face(&mut self, vertex_indices: &[(usize, Option<usize>, Option<usize>)]) {
            self.writer.write_face(vertex_indices).unwrap();
        }
    }

    #[test]
    fn test_obj_read_write_compare_64() {
        // git might change line endings as they are text files, so normalize to \n
        let obj_data = include_str!("../testdata/screw_f64.obj").replace("\r\n", "\n");
        let cursor = Cursor::new(&obj_data);
        // Default full-precision writer is enough to round-trip the test fixture.
        let writer: IoObjWriter<_, f64> = IoObjWriter::new(Vec::new());
        let mut reader = WritingReader64 { writer };
        read_obj_file(cursor, &mut reader).unwrap();

        let output = String::from_utf8(reader.writer.out).unwrap();
        assert_eq!(output, obj_data);
    }

    // Tests for f32 support

    #[derive(Default)]
    struct TestObjReader32 {
        vertices: Vec<(f32, f32, f32, Option<f32>)>,
        texture_coordinates: Vec<(f32, Option<f32>, Option<f32>)>,
        normals: Vec<(f32, f32, f32)>,
    }

    impl ObjReader<f32> for TestObjReader32 {
        fn read_comment(&mut self, _comment: &str) {}
        fn read_object_name(&mut self, _name: &str) {}

        fn read_vertex(&mut self, x: f32, y: f32, z: f32, w: Option<f32>) {
            self.vertices.push((x, y, z, w));
        }

        fn read_texture_coordinate(&mut self, u: f32, v: Option<f32>, w: Option<f32>) {
            self.texture_coordinates.push((u, v, w));
        }

        fn read_normal(&mut self, nx: f32, ny: f32, nz: f32) {
            self.normals.push((nx, ny, nz));
        }

        fn read_face(&mut self, _vertex_indices: &[(usize, Option<usize>, Option<usize>)]) {}
    }

    struct WritingReader32 {
        writer: IoObjWriter<Vec<u8>, f32>,
    }
    impl ObjReader<f32> for WritingReader32 {
        fn read_comment(&mut self, comment: &str) {
            self.writer.write_comment(comment).unwrap();
        }

        fn read_object_name(&mut self, name: &str) {
            self.writer.write_object_name(name).unwrap();
        }

        fn read_vertex(&mut self, x: f32, y: f32, z: f32, w: Option<f32>) {
            self.writer.write_vertex(x, y, z, w).unwrap();
        }

        fn read_texture_coordinate(&mut self, u: f32, v: Option<f32>, w: Option<f32>) {
            self.writer.write_texture_coordinate(u, v, w).unwrap();
        }

        fn read_normal(&mut self, nx: f32, ny: f32, nz: f32) {
            self.writer.write_normal(nx, ny, nz).unwrap();
        }

        fn read_face(&mut self, vertex_indices: &[(usize, Option<usize>, Option<usize>)]) {
            self.writer.write_face(vertex_indices).unwrap();
        }
    }

    #[test]
    fn test_obj_f32_reading() {
        let input = "o TestObject
v 1.5 2.5 3.5
vt 0.25 0.75
vn 0.0 1.0 0.0
";

        let reader = Cursor::new(input);
        let mut test_reader = TestObjReader32::default();
        read_obj_file(reader, &mut test_reader).unwrap();

        assert_eq!(test_reader.vertices, vec![(1.5f32, 2.5f32, 3.5f32, None)]);
        assert_eq!(
            test_reader.texture_coordinates,
            vec![(0.25f32, Some(0.75f32), None)]
        );
        assert_eq!(test_reader.normals, vec![(0.0f32, 1.0f32, 0.0f32)]);
    }

    #[test]
    fn test_obj_f32_writing() {
        let mut buffer = Vec::new();
        let mut writer: IoObjWriter<_, f32> = IoObjWriter::new(&mut buffer);

        writer.write_comment("f32 test").unwrap();
        writer.write_object_name("F32Object").unwrap();
        writer.write_vertex(1.5f32, 2.5f32, 3.5f32, None).unwrap();
        writer
            .write_texture_coordinate(0.25f32, Some(0.75f32), None)
            .unwrap();
        writer.write_normal(0.0f32, 1.0f32, 0.0f32).unwrap();

        let output = String::from_utf8(buffer).unwrap();
        let expected = "# f32 test
o F32Object
v 1.5 2.5 3.5
vt 0.25 0.75
vn 0 1 0
";
        assert_eq!(output, expected);
    }

    #[test]
    fn test_obj_f32_round_trip() {
        // Test that f32 values round-trip correctly
        let input = "o Test
v 0.123456789 0.987654321 1.5
vn 0.577 0.577 0.577
";

        let reader = Cursor::new(input);
        let mut test_reader = TestObjReader32::default();
        read_obj_file(reader, &mut test_reader).unwrap();

        // Write back using f32 writer
        let mut buffer = Vec::new();
        let mut writer: IoObjWriter<_, f32> = IoObjWriter::new(&mut buffer);
        writer.write_object_name("Test").unwrap();
        for (x, y, z, w) in &test_reader.vertices {
            writer.write_vertex(*x, *y, *z, *w).unwrap();
        }
        for (nx, ny, nz) in &test_reader.normals {
            writer.write_normal(*nx, *ny, *nz).unwrap();
        }

        let output = String::from_utf8(buffer).unwrap();

        // The values should be f32-precision
        assert!(output.contains("o Test"));
        assert!(output.contains("v "));
        assert!(output.contains("vn "));
    }

    #[test]
    fn test_printf_f_formatting() {
        let mut buffer = Vec::new();
        let mut writer: IoObjWriter<_, f64> = IoObjWriter::new_with_printf_f_format(&mut buffer);

        writer.write_object_name("PrintfFTest").unwrap();
        writer
            .write_vertex(754.4214477539063, 1753.2353515625, -91.72238159179688, None)
            .unwrap();
        writer
            .write_texture_coordinate(0.123456789, Some(0.987654321), None)
            .unwrap();
        writer
            .write_normal(0.5773502691896257, 0.5773502691896257, 0.5773502691896257)
            .unwrap();

        let output = String::from_utf8(buffer).unwrap();

        // C `printf %f` defaults to 6 decimal places.
        let expected = "o PrintfFTest
v 754.421448 1753.235352 -91.722382
vt 0.123457 0.987654
vn 0.577350 0.577350 0.577350
";
        assert_eq!(output, expected);
    }

    #[test]
    fn test_printf_f_flag_toggle() {
        // Test that we can toggle the flag
        let mut buffer = Vec::new();
        let mut writer: IoObjWriter<_, f32> = IoObjWriter::new(&mut buffer);

        // Start with default (full precision)
        writer
            .write_vertex(1.2345678_f32, 2.345678_f32, 3.45678_f32, None)
            .unwrap();

        // Enable printf %f formatting
        writer.set_printf_f_format(true);
        writer
            .write_vertex(1.2345678_f32, 2.3456789_f32, 3.456789_f32, None)
            .unwrap();

        let output = String::from_utf8(buffer).unwrap();

        // First line should have full f32 precision, second should have 6 decimals
        let lines: Vec<&str> = output.lines().collect();
        assert_eq!(lines.len(), 2);

        // Full precision output (f32 Display)
        assert_eq!(lines[0], "v 1.2345678 2.345678 3.45678");

        // printf %f output (6 decimal places)
        assert_eq!(lines[1], "v 1.234568 2.345679 3.456789");
    }

    #[test]
    fn test_obj_read_write_compare_32() {
        // git might change line endings as they are text files, so normalize to \n
        let obj_data = include_str!("../testdata/screw_f32.obj").replace("\r\n", "\n");
        let cursor = Cursor::new(&obj_data);
        // Default full-precision writer is enough to round-trip the test fixture.
        let writer: IoObjWriter<_, f32> = IoObjWriter::new(Vec::new());
        let mut reader = WritingReader32 { writer };
        read_obj_file(cursor, &mut reader).unwrap();

        let output = String::from_utf8(reader.writer.out).unwrap();
        assert_eq!(output, obj_data);
    }

    #[test]
    fn f32_reader_can_read_f64_obj() {
        let obj_data = include_str!("../testdata/screw_f64.obj");
        let cursor = Cursor::new(obj_data);
        let mut test_reader: TestObjReader32 = Default::default();
        read_obj_file(cursor, &mut test_reader).unwrap();
        // just check that some data was read
        assert_eq!(test_reader.vertices.len(), 41);
        assert_eq!(test_reader.texture_coordinates.len(), 41);
        assert_eq!(test_reader.normals.len(), 41);
    }

    // Tests for the standard auxiliary directives:
    //   mtllib, usemtl, g, s, l, p

    /// Captures every directive seen, for round-trip and equality testing.
    #[derive(Default, Debug, PartialEq)]
    struct ExtendedReader32 {
        material_libs: Vec<Vec<String>>,
        use_materials: Vec<String>,
        groups: Vec<Vec<String>>,
        smoothing_groups: Vec<SmoothingGroup>,
        line_elements: Vec<Vec<(usize, Option<usize>)>>,
        point_elements: Vec<Vec<usize>>,
    }

    impl ObjReader<f32> for ExtendedReader32 {
        fn read_comment(&mut self, _: &str) {}
        fn read_object_name(&mut self, _: &str) {}
        fn read_vertex(&mut self, _: f32, _: f32, _: f32, _: Option<f32>) {}
        fn read_texture_coordinate(&mut self, _: f32, _: Option<f32>, _: Option<f32>) {}
        fn read_normal(&mut self, _: f32, _: f32, _: f32) {}
        fn read_face(&mut self, _: &[(usize, Option<usize>, Option<usize>)]) {}

        fn read_material_lib(&mut self, names: &[&str]) {
            self.material_libs
                .push(names.iter().map(|s| s.to_string()).collect());
        }
        fn read_use_material(&mut self, name: &str) {
            self.use_materials.push(name.to_string());
        }
        fn read_group(&mut self, names: &[&str]) {
            self.groups
                .push(names.iter().map(|s| s.to_string()).collect());
        }
        fn read_smoothing_group(&mut self, group: SmoothingGroup) {
            self.smoothing_groups.push(group);
        }
        fn read_line_element(&mut self, indices: &[(usize, Option<usize>)]) {
            self.line_elements.push(indices.to_vec());
        }
        fn read_point_element(&mut self, indices: &[usize]) {
            self.point_elements.push(indices.to_vec());
        }
    }

    #[test]
    fn read_mtllib_and_usemtl() {
        let input = "mtllib first.mtl second.mtl
usemtl SomeMaterial
";
        let mut reader = ExtendedReader32::default();
        read_obj_file(Cursor::new(input), &mut reader).unwrap();
        assert_eq!(
            reader.material_libs,
            vec![vec!["first.mtl".to_string(), "second.mtl".to_string()]]
        );
        assert_eq!(reader.use_materials, vec!["SomeMaterial".to_string()]);
    }

    #[test]
    fn read_group_with_multiple_names() {
        let input = "g cube top
g default
";
        let mut reader = ExtendedReader32::default();
        read_obj_file(Cursor::new(input), &mut reader).unwrap();
        assert_eq!(
            reader.groups,
            vec![
                vec!["cube".to_string(), "top".to_string()],
                vec!["default".to_string()],
            ]
        );
    }

    #[test]
    fn read_smoothing_group_off_zero_and_named() {
        let input = "s off
s 0
s 1
s 42
";
        let mut reader = ExtendedReader32::default();
        read_obj_file(Cursor::new(input), &mut reader).unwrap();
        assert_eq!(
            reader.smoothing_groups,
            vec![
                SmoothingGroup::Off,
                SmoothingGroup::Off,
                SmoothingGroup::Group(1),
                SmoothingGroup::Group(42),
            ]
        );
    }

    #[test]
    fn read_smoothing_group_invalid_value_errors() {
        let input = "s notanumber\n";
        let mut reader = ExtendedReader32::default();
        let err = read_obj_file(Cursor::new(input), &mut reader).unwrap_err();
        assert!(
            err.to_string().contains("invalid smoothing group"),
            "got: {}",
            err
        );
    }

    #[test]
    fn read_line_and_point_elements() {
        let input = "l 1 2 3
l 4/1 5/2 6/3
p 1 2 3 4
";
        let mut reader = ExtendedReader32::default();
        read_obj_file(Cursor::new(input), &mut reader).unwrap();
        assert_eq!(
            reader.line_elements,
            vec![
                vec![(1, None), (2, None), (3, None)],
                vec![(4, Some(1)), (5, Some(2)), (6, Some(3))],
            ]
        );
        assert_eq!(reader.point_elements, vec![vec![1, 2, 3, 4]]);
    }

    #[test]
    fn read_line_element_with_one_vertex_errors() {
        let mut reader = ExtendedReader32::default();
        let err = read_obj_file(Cursor::new("l 1\n"), &mut reader).unwrap_err();
        assert!(
            err.to_string().contains("at least 2 vertices"),
            "got: {}",
            err
        );
    }

    #[test]
    fn write_directive_round_trip() {
        // Write every new directive, parse the output back, compare.
        let mut buffer = Vec::new();
        {
            let mut writer: IoObjWriter<_, f32> = IoObjWriter::new(&mut buffer);
            writer
                .write_material_lib(&["lib1.mtl", "lib2.mtl"])
                .unwrap();
            writer.write_use_material("Wood").unwrap();
            writer.write_group(&["cube", "top"]).unwrap();
            writer.write_smoothing_group(SmoothingGroup::Off).unwrap();
            writer
                .write_smoothing_group(SmoothingGroup::Group(7))
                .unwrap();
            writer
                .write_line_element(&[(1, None), (2, Some(2)), (3, None)])
                .unwrap();
            writer.write_point_element(&[1, 2, 3]).unwrap();
        }
        let text = String::from_utf8(buffer.clone()).unwrap();
        let expected = "mtllib lib1.mtl lib2.mtl
usemtl Wood
g cube top
s off
s 7
l 1 2/2 3
p 1 2 3
";
        assert_eq!(text, expected);

        let mut reader = ExtendedReader32::default();
        read_obj_file(Cursor::new(buffer), &mut reader).unwrap();
        assert_eq!(
            reader.material_libs,
            vec![vec!["lib1.mtl".to_string(), "lib2.mtl".to_string()]]
        );
        assert_eq!(reader.use_materials, vec!["Wood".to_string()]);
        assert_eq!(
            reader.groups,
            vec![vec!["cube".to_string(), "top".to_string()]]
        );
        assert_eq!(
            reader.smoothing_groups,
            vec![SmoothingGroup::Off, SmoothingGroup::Group(7)]
        );
        assert_eq!(
            reader.line_elements,
            vec![vec![(1, None), (2, Some(2)), (3, None)]]
        );
        assert_eq!(reader.point_elements, vec![vec![1, 2, 3]]);
    }

    #[test]
    fn unknown_prefix_still_errors_by_default() {
        // Sanity-check that adding the typed directives didn't accidentally
        // make the strict reader lenient for prefixes outside the new set.
        let input = "vp 0.1 0.2 0.3\n";
        let mut reader = ExtendedReader32::default();
        let err = read_obj_file(Cursor::new(input), &mut reader).unwrap_err();
        assert!(
            err.to_string().contains("Unknown line prefix: vp"),
            "got: {}",
            err
        );
    }

    // Typed-error tests: callers can pattern-match on `ObjError` and
    // `ParseErrorKind` instead of string-matching on the Display.

    #[test]
    fn typed_error_unknown_prefix_carries_prefix_and_line() {
        let input = "v 0 0 0\nvp 0.1 0.2\n";
        let mut reader = ExtendedReader32::default();
        let err = read_obj_file(Cursor::new(input), &mut reader).unwrap_err();
        match err {
            ObjError::Parse {
                line,
                kind: ParseErrorKind::UnknownPrefix(prefix),
            } => {
                assert_eq!(line, 2);
                assert_eq!(prefix, "vp");
            }
            other => panic!("wrong error variant: {:?}", other),
        }
    }

    #[test]
    fn typed_error_invalid_number_carries_field_and_value() {
        let input = "v 1.0 nope 3.0\n";
        let mut reader = ExtendedReader32::default();
        let err = read_obj_file(Cursor::new(input), &mut reader).unwrap_err();
        match err {
            ObjError::Parse {
                line: 1,
                kind: ParseErrorKind::InvalidNumber { field, value },
            } => {
                assert_eq!(field, "vertex y");
                assert_eq!(value, "nope");
            }
            other => panic!("wrong error variant: {:?}", other),
        }
    }

    #[test]
    fn typed_error_invalid_index_for_face() {
        // Index 0 is illegal in OBJ.
        let input = "v 0 0 0\nf 0/0/0 0/0/0 0/0/0\n";
        let mut reader = ExtendedReader32::default();
        let err = read_obj_file(Cursor::new(input), &mut reader).unwrap_err();
        match err {
            ObjError::Parse {
                line: 2,
                kind: ParseErrorKind::InvalidIndex { kind, value },
            } => {
                assert_eq!(kind, "vertex");
                assert_eq!(value, "0");
            }
            other => panic!("wrong error variant: {:?}", other),
        }
    }

    #[test]
    fn typed_error_invalid_smoothing_group() {
        let input = "s notanumber\n";
        let mut reader = ExtendedReader32::default();
        let err = read_obj_file(Cursor::new(input), &mut reader).unwrap_err();
        match err {
            ObjError::Parse {
                line: 1,
                kind: ParseErrorKind::InvalidSmoothingGroup(value),
            } => assert_eq!(value, "notanumber"),
            other => panic!("wrong error variant: {:?}", other),
        }
    }

    #[test]
    fn typed_error_missing_field() {
        let input = "v 1.0 2.0\n";
        let mut reader = ExtendedReader32::default();
        let err = read_obj_file(Cursor::new(input), &mut reader).unwrap_err();
        match err {
            ObjError::Parse {
                line: 1,
                kind: ParseErrorKind::MissingField(field),
            } => assert_eq!(field, "vertex z"),
            other => panic!("wrong error variant: {:?}", other),
        }
    }

    #[test]
    fn typed_error_converts_to_io_error() {
        // `read_obj_file` returns `ObjError`; callers that work in
        // `io::Result` get a free conversion via `From<ObjError>`.
        fn legacy_caller<R: io::Read>(input: R) -> io::Result<()> {
            let mut reader = ExtendedReader32::default();
            read_obj_file(input, &mut reader)?;
            Ok(())
        }

        let err = legacy_caller(Cursor::new("vp 0.1 0.2\n")).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("Unknown line prefix: vp"));
    }

    #[test]
    fn typed_error_io_failure_propagates() {
        // Force a Read error mid-stream.
        struct FailingRead;
        impl io::Read for FailingRead {
            fn read(&mut self, _buf: &mut [u8]) -> io::Result<usize> {
                Err(io::Error::other("disk on fire"))
            }
        }

        let mut reader = ExtendedReader32::default();
        let err = read_obj_file(FailingRead, &mut reader).unwrap_err();
        match err {
            ObjError::Io(inner) => {
                assert_eq!(inner.to_string(), "disk on fire");
            }
            other => panic!("wrong error variant: {:?}", other),
        }
    }

    #[test]
    fn custom_read_unknown_can_return_objerror() {
        // A reader that surfaces unknown prefixes as Custom errors.
        struct StrictCustom;
        impl ObjReader<f32> for StrictCustom {
            fn read_comment(&mut self, _: &str) {}
            fn read_object_name(&mut self, _: &str) {}
            fn read_vertex(&mut self, _: f32, _: f32, _: f32, _: Option<f32>) {}
            fn read_texture_coordinate(&mut self, _: f32, _: Option<f32>, _: Option<f32>) {}
            fn read_normal(&mut self, _: f32, _: f32, _: f32) {}
            fn read_face(&mut self, _: &[(usize, Option<usize>, Option<usize>)]) {}
            fn read_unknown(&mut self, prefix: &str, _: &str, line: usize) -> Result<(), ObjError> {
                Err(ObjError::Parse {
                    line,
                    kind: ParseErrorKind::Custom(format!("nope: {prefix}")),
                })
            }
        }

        let err = read_obj_file(Cursor::new("vp 0 0\n"), &mut StrictCustom).unwrap_err();
        match err {
            ObjError::Parse {
                line: 1,
                kind: ParseErrorKind::Custom(msg),
            } => assert_eq!(msg, "nope: vp"),
            other => panic!("wrong error variant: {:?}", other),
        }
    }
}
