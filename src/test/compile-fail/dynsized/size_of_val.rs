// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Ensure `size_of_val` / `align_of_val` can't be used on !DynSized types

#![feature(extern_types)]

use std::mem::{size_of_val, align_of_val};

extern {
    type A;
}

fn main() {
    let x: &A = unsafe {
        &*(1usize as *const A)
    };

    size_of_val(x); //~ERROR the trait bound `A: std::marker::DynSized` is not satisfied

    align_of_val(x); //~ERROR the trait bound `A: std::marker::DynSized` is not satisfied
}
