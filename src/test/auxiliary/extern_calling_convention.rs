// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Make sure Rust generates the correct calling convention for extern
// functions.

extern crate debug;

#[inline(never)]
#[cfg(target_arch = "x86_64")]
pub extern "win64" fn foo(a: int, b: int, c: int, d: int) {
    assert!(a == 1);
    assert!(b == 2);
    assert!(c == 3);
    assert!(d == 4);

    println!("a: {:?}, b: {:?}, c: {:?}, d: {:?}",
             a, b, c, d)
}

#[inline(never)]
#[cfg(target_arch = "x86")]
#[cfg(target_arch = "arm")]
pub extern fn foo(a: int, b: int, c: int, d: int) {
    assert!(a == 1);
    assert!(b == 2);
    assert!(c == 3);
    assert!(d == 4);

    println!("a: {:?}, b: {:?}, c: {:?}, d: {:?}",
             a, b, c, d)
}
