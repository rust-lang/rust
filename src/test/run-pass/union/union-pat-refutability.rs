// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(untagged_unions)]

#[repr(u32)]
enum Tag { I, F }

#[repr(C)]
union U {
    i: i32,
    f: f32,
}

#[repr(C)]
struct Value {
    tag: Tag,
    u: U,
}

fn is_zero(v: Value) -> bool {
    unsafe {
        match v {
            Value { tag: Tag::I, u: U { i: 0 } } => true,
            Value { tag: Tag::F, u: U { f: 0.0 } } => true,
            _ => false,
        }
    }
}

union W {
    a: u8,
    b: u8,
}

fn refut(w: W) {
    unsafe {
        match w {
            W { a: 10 } => {
                panic!();
            }
            W { b } => {
                assert_eq!(b, 11);
            }
        }
    }
}

fn main() {
    let v = Value { tag: Tag::I, u: U { i: 1 } };
    assert_eq!(is_zero(v), false);

    let w = W { a: 11 };
    refut(w);
}
