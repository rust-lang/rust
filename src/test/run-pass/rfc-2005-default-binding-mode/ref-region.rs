// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(match_default_bindings)]

fn foo<'a, 'b>(x: &'a &'b Option<u32>) -> &'a u32 {
    let x: &'a &'a Option<u32> = x;
    match x {
        Some(r) => {
            let _: &u32 = r;
            r
        },
        &None => panic!(),
    }
}

pub fn main() {
    let x = Some(5);
    foo(&&x);
}
