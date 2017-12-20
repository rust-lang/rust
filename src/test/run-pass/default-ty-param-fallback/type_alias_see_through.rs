// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(default_type_parameter_fallback)]

type Strang = &'static str;

fn main() {
    let a = None;
    func1(a);
    func2(a);
}

// Defaults on fns take precedence.
fn func1<P = &'static str>(_: Option<P>) {}
fn func2<P = Strang>(_: Option<P>) {}
