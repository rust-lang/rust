// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Check that when making a ref mut binding with type `&mut T`, the
// type `T` must match precisely the type `U` of the value being
// matched, and in particular cannot be some supertype of `U`. Issue
// #23116. This test focuses on a `let`.

#![allow(dead_code)]
struct S<'b>(&'b i32);
impl<'b> S<'b> {
    fn bar<'a>(&'a mut self) -> &'a mut &'a i32 {
        let ref mut x = self.0;
        x //~ ERROR mismatched types
    }
}

fn main() {}
