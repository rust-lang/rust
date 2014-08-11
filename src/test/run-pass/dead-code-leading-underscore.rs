// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![deny(dead_code)]

static _X: uint = 0;

fn _foo() {}

struct _Y {
    _z: uint
}

enum _Z {}

impl _Y {
    fn _bar() {}
}

type _A = int;

mod _bar {
    fn _qux() {}
}

extern {
    #[link_name = "abort"]
    fn _abort() -> !;
}

pub fn main() {}
