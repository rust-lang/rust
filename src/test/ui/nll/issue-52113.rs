// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
//

#![allow(warnings)]
#![feature(nll)]

trait Bazinga { }
impl<F> Bazinga for F { }

fn produce1<'a>(data: &'a u32) -> impl Bazinga + 'a {
    let x = move || {
        let _data: &'a u32 = data;
    };
    x
}

fn produce2<'a>(data: &'a mut Vec<&'a u32>, value: &'a u32) -> impl Bazinga + 'a {
    let x = move || {
        let value: &'a u32 = value;
        data.push(value);
    };
    x
}


fn produce3<'a, 'b: 'a>(data: &'a mut Vec<&'a u32>, value: &'b u32) -> impl Bazinga + 'a {
    let x = move || {
        let value: &'a u32 = value;
        data.push(value);
    };
    x
}

fn produce_err<'a, 'b: 'a>(data: &'b mut Vec<&'b u32>, value: &'a u32) -> impl Bazinga + 'b {
    let x = move || { //~ ERROR unsatisfied lifetime constraints
        let value: &'a u32 = value;
        data.push(value);
    };
    x
}

fn main() { }
