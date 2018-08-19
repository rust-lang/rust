// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(generic_associated_types)]

//FIXME(#44265): The lifetime shadowing and type parameter shadowing
// should cause an error. Now it compiles (errorneously) and this will be addressed
// by a future PR. Then remove the following:
// compile-pass

trait Shadow<'a> {
    type Bar<'a>; // Error: shadowed lifetime
}

trait NoShadow<'a> {
    type Bar<'b>; // OK
}

impl<'a> NoShadow<'a> for &'a u32 {
    type Bar<'a> = i32; // Error: shadowed lifetime
}

trait ShadowT<T> {
    type Bar<T>; // Error: shadowed type parameter
}

trait NoShadowT<T> {
    type Bar<U>; // OK
}

impl<T> NoShadowT<T> for Option<T> {
    type Bar<T> = i32; // Error: shadowed type parameter
}

fn main() {}
