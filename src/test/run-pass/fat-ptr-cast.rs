// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(core)]

use std::mem;
use std::raw;

trait Foo {
    fn foo(&self) {}
}

struct Bar;

impl Foo for Bar {}

fn main() {
    // Test we can turn a fat pointer to array back into a thin pointer.
    let a: *const [i32] = &[1, 2, 3];
    let b = a as *const [i32; 2];
    unsafe {
        assert!(*b == [1, 2]);
    }

    // Test conversion to an address (usize).
    let a: *const [i32; 3] = &[1, 2, 3];
    let b: *const [i32] = a;
    assert!(a as usize == b as *const () as usize);

    // And conversion to a void pointer/address for trait objects too.
    let a: *mut Foo = &mut Bar;
    let b = a as *mut ();
    let c = a as *const () as usize;
    let d = unsafe {
        let r: raw::TraitObject = mem::transmute(a);
        r.data
    };

    assert!(b == d);
    assert!(c == d as usize);

}
