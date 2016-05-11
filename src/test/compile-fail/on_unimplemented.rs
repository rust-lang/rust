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

#[rustc_on_unimplemented = "a isize is required to index into a slice"]
impl Index<isize> for [i32] {
    type Output = i32;
    fn index(&self, index: isize) -> &i32 {
        &self[index as usize]
    }
}

#[rustc_on_unimplemented = "a usize is required to index into a slice"]
impl Index<usize> for [i32] {
    type Output = i32;
    fn index(&self, index: usize) -> &i32 {
        &self[index]
    }
}

trait Foo<A, B> {
    fn f(&self, a: &A, b: &B);
}

#[rustc_on_unimplemented = "two i32 Foo trait takes"]
impl Foo<i32, i32> for [i32] {
    fn f(&self, a: &i32, b: &i32) {}
}

#[rustc_on_unimplemented = "two u32 Foo trait takes"]
impl Foo<u32, u32> for [i32] {
    fn f(&self, a: &u32, b: &u32) {}
}

#[rustc_error]
fn main() {
    Index::<u32>::index(&[1, 2, 3] as &[i32], 2u32); //~ ERROR E0277
                                                     //~| NOTE a usize is required
                                                     //~| NOTE required by
    Index::<i32>::index(&[1, 2, 3] as &[i32], 2i32); //~ ERROR E0277
                                                     //~| NOTE a isize is required
                                                     //~| NOTE required by

    Foo::<usize, usize>::f(&[1, 2, 3] as &[i32], &2usize, &2usize); //~ ERROR E0277
                                                                    //~| NOTE two u32 Foo trait
                                                                    //~| NOTE required by
}
