// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::mutable::Mut;
use std::rc::Rc;

fn o<T: Send>(_: &T) {}
fn c<T: Freeze>(_: &T) {}

fn main() {
    let x = Rc::from_send(Mut::new(0));
    o(&x); //~ ERROR instantiating a type parameter with an incompatible type `std::rc::Rc<std::mutable::Mut<int>>`, which does not fulfill `Send`
    c(&x); //~ ERROR instantiating a type parameter with an incompatible type `std::rc::Rc<std::mutable::Mut<int>>`, which does not fulfill `Freeze`
}
