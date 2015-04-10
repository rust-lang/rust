// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern crate libc;

fn main() {
    let foo: *mut libc::c_void;
    let cb: &mut Fn() = unsafe {
        &mut *(foo as *mut Fn())
        //~^ ERROR use of possibly uninitialized variable: `foo`
    };
}
