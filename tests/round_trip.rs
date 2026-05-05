//! Integration tests that exercise the public API against real OBJ fixtures.

use pretty_assertions::assert_eq;
use std::io::Cursor;
use wavefront_obj_io::{IoObjWriter, ObjReader, ObjWriter, read_obj_file};

type Face = Vec<(usize, Option<usize>, Option<usize>)>;

#[derive(Default)]
struct CollectingReader64 {
    comments: Vec<String>,
    names: Vec<String>,
    vertices: Vec<(f64, f64, f64, Option<f64>)>,
    texture_coordinates: Vec<(f64, Option<f64>, Option<f64>)>,
    normals: Vec<(f64, f64, f64)>,
    faces: Vec<Face>,
}

impl ObjReader for CollectingReader64 {
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

#[derive(Default)]
struct CollectingReader32 {
    vertices: Vec<(f32, f32, f32, Option<f32>)>,
    texture_coordinates: Vec<(f32, Option<f32>, Option<f32>)>,
    normals: Vec<(f32, f32, f32)>,
}

impl ObjReader<f32> for CollectingReader32 {
    fn read_comment(&mut self, _: &str) {}
    fn read_object_name(&mut self, _: &str) {}
    fn read_vertex(&mut self, x: f32, y: f32, z: f32, w: Option<f32>) {
        self.vertices.push((x, y, z, w));
    }
    fn read_texture_coordinate(&mut self, u: f32, v: Option<f32>, w: Option<f32>) {
        self.texture_coordinates.push((u, v, w));
    }
    fn read_normal(&mut self, nx: f32, ny: f32, nz: f32) {
        self.normals.push((nx, ny, nz));
    }
    fn read_face(&mut self, _: &[(usize, Option<usize>, Option<usize>)]) {}
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
fn read_screw_f64_collects_all_directives() {
    let obj_data = include_str!("fixtures/screw_f64.obj");
    let mut reader = CollectingReader64::default();
    read_obj_file(Cursor::new(obj_data), &mut reader).unwrap();
    assert_eq!(reader.comments.len(), 3);
    assert_eq!(reader.names.len(), 1);
    assert_eq!(reader.vertices.len(), 41);
    assert_eq!(reader.texture_coordinates.len(), 41);
    assert_eq!(reader.normals.len(), 41);
    assert_eq!(reader.faces.len(), 48);
}

#[test]
fn round_trip_screw_f64_is_byte_identical() {
    // git might rewrite line endings; normalize to \n.
    let obj_data = include_str!("fixtures/screw_f64.obj").replace("\r\n", "\n");
    let writer: IoObjWriter<_, f64> = IoObjWriter::new(Vec::new());
    let mut reader = WritingReader64 { writer };
    read_obj_file(Cursor::new(&obj_data), &mut reader).unwrap();

    let output = String::from_utf8(reader.writer.into_inner()).unwrap();
    assert_eq!(output, obj_data);
}

#[test]
fn round_trip_screw_f32_is_byte_identical() {
    let obj_data = include_str!("fixtures/screw_f32.obj").replace("\r\n", "\n");
    let writer: IoObjWriter<_, f32> = IoObjWriter::new(Vec::new());
    let mut reader = WritingReader32 { writer };
    read_obj_file(Cursor::new(&obj_data), &mut reader).unwrap();

    let output = String::from_utf8(reader.writer.into_inner()).unwrap();
    assert_eq!(output, obj_data);
}

#[test]
fn f32_reader_can_read_f64_fixture() {
    let obj_data = include_str!("fixtures/screw_f64.obj");
    let mut reader = CollectingReader32::default();
    read_obj_file(Cursor::new(obj_data), &mut reader).unwrap();
    assert_eq!(reader.vertices.len(), 41);
    assert_eq!(reader.texture_coordinates.len(), 41);
    assert_eq!(reader.normals.len(), 41);
}
