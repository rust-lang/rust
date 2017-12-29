// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(rustc_attrs)]

use std::mem;

trait Trait1<T> {}
trait Trait2<'a> {
  type Ty;
}

fn _ice(param: Box<for <'a> Trait1<<() as Trait2<'a>>::Ty>>) {
    let _e: (usize, usize) = unsafe{mem::transmute(param)};
}

trait Lifetime<'a> {
    type Out;
}
impl<'a> Lifetime<'a> for () {
    type Out = &'a ();
}
fn foo<'a>(x: &'a ()) -> <() as Lifetime<'a>>::Out {
    x
}

fn takes_lifetime(_f: for<'a> fn(&'a ()) -> <() as Lifetime<'a>>::Out) {
}

#[rustc_error]
fn main() { //~ ERROR compilation successful
    takes_lifetime(foo);
}
