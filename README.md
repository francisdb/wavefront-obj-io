# wavefront-obj-io

[![Crates.io](https://img.shields.io/crates/v/wavefront-obj-io.svg)](https://crates.io/crates/wavefront-obj-io)
[![Docs.rs](https://docs.rs/wavefront-obj-io/badge.svg)](https://docs.rs/wavefront-obj-io)

Streaming, callback-based Wavefront OBJ and MTL reader and writer in Rust.
For round-trip pipelines and large files.

## When to use this crate

- You want to process OBJ files **without** allocating a `Mesh` struct -
  e.g. push vertices straight into a GPU buffer.
- You want to **round-trip** an OBJ file (read then write back) without
  losing fidelity.
- You're building an OBJ format converter / preprocessor / validator.
- You need to handle files too large to comfortably fit in memory.

## When *not* to use this crate

If you just want `let mesh = load(path)?`, use
[`tobj`](https://crates.io/crates/tobj). It is the right tool for that job
and this crate intentionally does not compete with it.

## Example

```rust
use wavefront_obj_io::{ObjReader, read_obj_file};
use std::io::Cursor;

#[derive(Default)]
struct CountVertices(usize);

impl ObjReader<f32> for CountVertices {
    fn read_comment(&mut self, _: &str) {}
    fn read_object_name(&mut self, _: &str) {}
    fn read_vertex(&mut self, _: f32, _: f32, _: f32, _: Option<f32>) {
        self.0 += 1;
    }
    fn read_texture_coordinate(&mut self, _: f32, _: Option<f32>, _: Option<f32>) {}
    fn read_normal(&mut self, _: f32, _: f32, _: f32) {}
    fn read_face(&mut self, _: &[(usize, Option<usize>, Option<usize>)]) {}
}

let obj = "v 0 0 0\nv 1 0 0\nv 0 1 0\n";
let mut counter = CountVertices::default();
read_obj_file(Cursor::new(obj), &mut counter).unwrap();
assert_eq!(counter.0, 3);
```

## Features

- Streaming, SAX-style callback API on `ObjReader` / `ObjWriter` traits.
- Configurable float precision (`f32` or `f64`) via the `ObjFloat` generic.
- Standard OBJ directive coverage: `v`, `vt`, `vn`, `f`, `o`, `#`, `mtllib`,
  `usemtl`, `g`, `s`, `l`, `p`.
- MTL directive coverage: `newmtl`, `Ka`, `Kd`, `Ks`, `Ke`, `Ns`, `Ni`,
  `d`, `Tr`, `illum`, and the `map_*` / `bump` / `disp` / `decal` / `refl`
  texture-map family.
- Strict-by-default `read_unknown` callback - opt in to lenient parsing
  for NURBS / display / vendor directives.
- Typed `ObjError` with structured `ParseErrorKind` for pattern-matching;
  bidirectional `From` between `ObjError` and `io::Error`.
- WASM-compatible, no platform-specific dependencies.

## Documentation

<https://docs.rs/wavefront-obj-io>

## Development

This project uses [Conventional Commits](https://www.conventionalcommits.org/)
for commit messages, and `rustfmt` + `clippy` for formatting and linting.

```bash
cargo fmt
cargo clippy --all-targets -- -D warnings
```

## Running the tests

```bash
cargo test
```

### WASM tests for server-side WASM (wasmtime)

```bash
# Install the target and wasmtime (one-time setup)
rustup target add wasm32-wasip1
cargo install wasmtime-cli

cargo test --target wasm32-wasip1
```

### WASM build for browser

```bash
rustup target add wasm32-unknown-unknown

cargo build --target wasm32-unknown-unknown
```

## License

MIT
