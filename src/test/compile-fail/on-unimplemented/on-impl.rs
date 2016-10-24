// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test if the on_unimplemented message override works

#![feature(on_unimplemented)]
#![feature(rustc_attrs)]

#[rustc_on_unimplemented = "invalid"]
trait Index<Idx: ?Sized> {
    type Output: ?Sized;
    fn index(&self, index: Idx) -> &Self::Output;
}

#[rustc_on_unimplemented = "a usize is required to index into a slice"]
impl Index<usize> for [i32] {
    type Output = i32;
    fn index(&self, index: usize) -> &i32 {
        &self[index]
    }
}

#[rustc_error]
fn main() {
    Index::<u32>::index(&[1, 2, 3] as &[i32], 2u32);
    //~^ ERROR E0277
    //~| NOTE the trait `Index<u32>` is not implemented for `[i32]`
    //~| NOTE a usize is required
    //~| NOTE required by
}
