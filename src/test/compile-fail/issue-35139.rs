// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::fmt;

pub trait MethodType {
    type GetProp: ?Sized;
}

pub struct MTFn;

impl<'a> MethodType for MTFn { //~ ERROR E0207
                               //~| NOTE unconstrained lifetime parameter
    type GetProp = fmt::Debug + 'a;
}

fn bad(a: Box<<MTFn as MethodType>::GetProp>) -> Box<fmt::Debug+'static> {
    a
}

fn dangling(a: &str) -> Box<fmt::Debug> {
    bad(Box::new(a))
}

fn main() {
    let mut s = "hello".to_string();
    let x = dangling(&s);
    s = String::new();
    println!("{:?}", x);
}
