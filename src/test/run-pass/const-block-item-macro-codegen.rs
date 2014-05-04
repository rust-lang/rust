// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// General test that function items in static blocks
// can be generated with a macro.

#![feature(macro_rules)]

struct MyType {
    desc: &'static str,
    data: uint,
    code: fn(uint, uint) -> uint
}

impl MyType {
    fn eval(&self, a: uint) -> uint {
        (self.code)(self.data, a)
    }
}

macro_rules! codegen {
    ($e:expr, $v:expr) => {
        {
            fn generated(a: uint, b: uint) -> uint {
                a - ($e * b)
            }
            MyType {
                desc: "test",
                data: $v,
                code: generated
            }
        }
    }
}

static GENERATED_CODE_1: MyType = codegen!(2, 100);
static GENERATED_CODE_2: MyType = codegen!(5, 1000);

pub fn main() {
    assert_eq!(GENERATED_CODE_1.eval(10), 80);
    assert_eq!(GENERATED_CODE_2.eval(100), 500);
}
