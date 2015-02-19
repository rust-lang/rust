// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

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
