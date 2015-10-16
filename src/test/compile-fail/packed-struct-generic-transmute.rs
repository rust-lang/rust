// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This assumes the packed and non-packed structs are different sizes.

// the error points to the start of the file, not the line with the
// transmute

// error-pattern: transmute called with differently sized types

use std::mem;

#[repr(packed)]
struct Foo<T,S> {
    bar: T,
    baz: S
}

struct Oof<T, S> {
    rab: T,
    zab: S
}

fn main() {
    let foo = Foo { bar: [1u8, 2, 3, 4, 5], baz: 10i32 };
    unsafe {
        let oof: Oof<[u8; 5], i32> = mem::transmute(foo);
        println!("{:?} {:?}", &oof.rab[..], oof.zab);
    }
}
