// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::ops::{Index, IndexMut};

struct S;
struct H;

impl S {
    fn f(&mut self) {}
}

impl Index<u32> for H {
    type Output = S;
    fn index(&self, index: u32) -> &S {
        unimplemented!()
    }
}

impl IndexMut<u32> for H {
    fn index_mut(&mut self, index: u32) -> &mut S {
        unimplemented!()
    }
}

fn main() {
    H["?"].f(); //~ ERROR mismatched types
}
