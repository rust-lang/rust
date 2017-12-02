// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub trait Parameters { type SelfRef; }

struct RP<'a> { _marker: std::marker::PhantomData<&'a ()> }
struct BP;

impl<'a> Parameters for RP<'a> { type SelfRef = &'a X<RP<'a>>; }
impl Parameters for BP { type SelfRef = Box<X<BP>>; }

pub struct Y;
pub enum X<P: Parameters> {
    Nothing,
    SameAgain(P::SelfRef, Y)
}

fn main() {
    let bnil: Box<X<BP>> = Box::new(X::Nothing);
    let bx: Box<X<BP>> = Box::new(X::SameAgain(bnil, Y));
    let rnil: X<RP> = X::Nothing;
    let rx: X<RP> = X::SameAgain(&rnil, Y);
}
