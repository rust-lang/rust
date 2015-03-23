// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This is a regression test for something that only came up while
// attempting to bootstrap libsyntax; it is adapted from
// `syntax::ext::tt::generic_extension`.

// pretty-expanded FIXME #23616

pub struct E<'a> {
    pub f: &'a u8,
}
impl<'b> E<'b> {
    pub fn m(&self) -> &'b u8 { self.f }
}

pub struct P<'c> {
    pub g: &'c u8,
}
pub trait M {
    fn n(&self) -> u8;
}
impl<'d> M for P<'d> {
    fn n(&self) -> u8 { *self.g }
}

fn extension<'e>(x: &'e E<'e>) -> Box<M+'e> {
    loop {
        let p = P { g: x.m() };
        return Box::new(p) as Box<M+'e>;
    }
}

fn main() {
    let w = E { f: &10 };
    let o = extension(&w);
    assert_eq!(o.n(), 10);
}
