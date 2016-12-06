// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


#[derive(Copy, Clone)]
pub struct S {
    x: u64,
    y: u64,
    z: u64,
}

#[link(name = "rust_test_helpers", kind = "static")]
extern {
    pub fn get_x(x: S) -> u64;
    pub fn get_y(x: S) -> u64;
    pub fn get_z(x: S) -> u64;
}

#[inline(never)]
fn indirect_call(func: unsafe extern fn(s: S) -> u64, s: S) -> u64 {
    unsafe {
        func(s)
    }
}

fn main() {
    let s = S { x: 1, y: 2, z: 3 };
    assert_eq!(s.x, indirect_call(get_x, s));
    assert_eq!(s.y, indirect_call(get_y, s));
    assert_eq!(s.z, indirect_call(get_z, s));
}
