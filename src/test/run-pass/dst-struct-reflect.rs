// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// FIXME(15049) Re-enable this test.
// ignore-test
// Test that structs with unsized fields work with {:?} reflection.

extern crate debug;

struct Fat<Sized? T> {
    f1: int,
    f2: &'static str,
    ptr: T
}

// x is a fat pointer
fn reflect(x: &Fat<[int]>, cmp: &str) {
    // Don't test this result because reflecting unsized fields is undefined for now.
    let _s = format!("{:?}", x);
    let s = format!("{:?}", &x.ptr);
    assert!(s == cmp.to_string())

    println!("{:?}", x);
    println!("{:?}", &x.ptr);
}

fn reflect_0(x: &Fat<[int]>) {
    let _s = format!("{:?}", x.ptr[0]);
    println!("{:?}", x.ptr[0]);
}

pub fn main() {
    // With a vec of ints.
    let f1 = Fat { f1: 5, f2: "some str", ptr: [1, 2, 3] };
    reflect(&f1, "&[1, 2, 3]");
    reflect_0(&f1);
    let f2 = &f1;
    reflect(f2, "&[1, 2, 3]");
    reflect_0(f2);
    let f3: &Fat<[int]> = f2;
    reflect(f3, "&[1, 2, 3]");
    reflect_0(f3);
    let f4: &Fat<[int]> = &f1;
    reflect(f4, "&[1, 2, 3]");
    reflect_0(f4);
    let f5: &Fat<[int]> = &Fat { f1: 5, f2: "some str", ptr: [1, 2, 3] };
    reflect(f5, "&[1, 2, 3]");
    reflect_0(f5);

    // Zero size vec.
    let f5: &Fat<[int]> = &Fat { f1: 5, f2: "some str", ptr: [] };
    reflect(f5, "&[]");
}

