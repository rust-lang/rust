// build-pass (FIXME(62277): could be check-pass?)
#![allow(dead_code)]

trait MatrixRow { fn dummy(&self) { }}

struct Mat;

impl<'a> MatrixRow for &'a Mat {}

struct Rows<M: MatrixRow> {
    mat: M,
}

impl<'a> Iterator for Rows<&'a Mat> {
    type Item = ();

    fn next(&mut self) -> Option<()> {
        unimplemented!()
    }
}

fn main() {}
