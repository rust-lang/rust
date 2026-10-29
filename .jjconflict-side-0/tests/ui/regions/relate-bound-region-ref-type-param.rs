//! Regression test for <https://github.com/rust-lang/rust/issues/16596>.

//@ check-pass
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
