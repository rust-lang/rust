// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern mod extra;

fn o<T: Owned>(_: &T) {}
fn c<T: Const>(_: &T) {}

fn main() {
    let x = extra::rc::rc_mut_from_owned(0);
    o(&x); //~ ERROR instantiating a type parameter with an incompatible type `extra::rc::RcMut<int>`, which does not fulfill `Owned`
    c(&x); //~ ERROR instantiating a type parameter with an incompatible type `extra::rc::RcMut<int>`, which does not fulfill `Const`
}
