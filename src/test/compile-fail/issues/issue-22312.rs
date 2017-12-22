// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::ops::Index;

pub trait Array2D: Index<usize> {
    fn rows(&self) -> usize;
    fn columns(&self) -> usize;
    fn get<'a>(&'a self, y: usize, x: usize) -> Option<&'a <Self as Index<usize>>::Output> {
        if y >= self.rows() || x >= self.columns() {
            return None;
        }
        let i = y * self.columns() + x;
        let indexer = &(*self as &Index<usize, Output = <Self as Index<usize>>::Output>);
        //~^ERROR non-primitive cast
        Some(indexer.index(i))
    }
}

fn main() {}
