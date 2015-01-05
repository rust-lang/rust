// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Uncovered during work on new scoping rules for safe destructors
// as an important use case to support properly.

pub struct E<'a> {
    pub f: &'a uint,
}
impl<'b> E<'b> {
    pub fn m(&self) -> &'b uint { self.f }
}

pub struct P<'c> {
    pub g: &'c uint,
}
pub trait M {
    fn n(&self) -> uint;
}
impl<'d> M for P<'d> {
    fn n(&self) -> uint { *self.g }
}

fn extension<'e>(x: &'e E<'e>) -> Box<M+'e> {
    loop {
        let p = P { g: x.m() };
        return box p as Box<M+'e>;
    }
}

fn main() {
    let w = E { f: &10u };
    let o = extension(&w);
    assert_eq!(o.n(), 10u);
}
