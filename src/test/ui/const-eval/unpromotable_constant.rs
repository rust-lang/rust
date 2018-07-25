// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(const_err)]

// compile-pass

union Bad {
    usize: usize,
    ptr: &'static u32,
}

// the constant does something "unconst", so promotion is allowed to fail
const FOO: usize = unsafe {
    Bad { ptr: &1 }.usize
};


fn main() {
    let x: &'static usize = &FOO;
    let y: &'static usize = &(FOO % 42);
}
