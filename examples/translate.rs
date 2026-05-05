//! Streaming modify-copy: read OBJ from stdin, translate every vertex by a
//! fixed offset, write the result to stdout. Faces, normals, and texture
//! coordinates pass through unchanged.
//!
//!   cargo run --example translate < input.obj > output.obj

use std::io::{self, BufWriter, Write};
use wavefront_obj_io::{
    IoObjWriter, ObjError, ObjReader, ObjWriter, SmoothingGroup, read_obj_file,
};

const OFFSET: (f64, f64, f64) = (1.0, 2.0, 3.0);

struct Translate<W: Write> {
    writer: IoObjWriter<W, f64>,
}

impl<W: Write> ObjReader<f64> for Translate<W> {
    fn read_comment(&mut self, c: &str) {
        self.writer.write_comment(c).unwrap();
    }
    fn read_object_name(&mut self, n: &str) {
        self.writer.write_object_name(n).unwrap();
    }
    fn read_vertex(&mut self, x: f64, y: f64, z: f64, w: Option<f64>) {
        self.writer
            .write_vertex(x + OFFSET.0, y + OFFSET.1, z + OFFSET.2, w)
            .unwrap();
    }
    fn read_texture_coordinate(&mut self, u: f64, v: Option<f64>, w: Option<f64>) {
        self.writer.write_texture_coordinate(u, v, w).unwrap();
    }
    fn read_normal(&mut self, nx: f64, ny: f64, nz: f64) {
        self.writer.write_normal(nx, ny, nz).unwrap();
    }
    fn read_face(&mut self, idx: &[(usize, Option<usize>, Option<usize>)]) {
        self.writer.write_face(idx).unwrap();
    }
    fn read_material_lib(&mut self, names: &[&str]) {
        self.writer.write_material_lib(names).unwrap();
    }
    fn read_use_material(&mut self, name: &str) {
        self.writer.write_use_material(name).unwrap();
    }
    fn read_group(&mut self, names: &[&str]) {
        self.writer.write_group(names).unwrap();
    }
    fn read_smoothing_group(&mut self, g: SmoothingGroup) {
        self.writer.write_smoothing_group(g).unwrap();
    }
    fn read_line_element(&mut self, indices: &[(usize, Option<usize>)]) {
        self.writer.write_line_element(indices).unwrap();
    }
    fn read_point_element(&mut self, indices: &[usize]) {
        self.writer.write_point_element(indices).unwrap();
    }
}

fn main() -> Result<(), ObjError> {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut filter = Translate {
        writer: IoObjWriter::new(BufWriter::new(stdout.lock())),
    };
    read_obj_file(stdin.lock(), &mut filter)?;
    Ok(())
}
