// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Ensure !DynSized fields can't be used in tuples, even as the last field.

#![feature(extern_types)]
#![feature(dynsized)]

use std::marker::DynSized;

extern {
    type foo;
}

fn baz<T: ?DynSized>() {
    let x: &(u8, foo); //~ERROR the trait bound `foo: std::marker::DynSized` is not satisfied

    let y: &(u8, T); //~ERROR the trait bound `T: std::marker::DynSized` is not satisfied
}

fn main() { }
