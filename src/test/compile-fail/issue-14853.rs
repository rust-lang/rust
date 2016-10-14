// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::fmt::Debug;

trait Str {}

trait Something: Sized {
    fn yay<T: Debug>(_: Option<Self>, thing: &[T]);
}

struct X { data: u32 }

impl Something for X {
    fn yay<T: Str>(_:Option<X>, thing: &[T]) {
    //~^ ERROR E0276
    }
}

fn main() {
    let arr = &["one", "two", "three"];
    println!("{:?}", Something::yay(None::<X>, arr));
}
