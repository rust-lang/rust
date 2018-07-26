// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub trait Foo {
    type Associated;
}

pub struct X;
pub struct Y;


impl Foo for X {
    type Associated = ();
}

impl Foo for Y {
    type Associated = ();
}

impl X {
    pub fn returns_sized<'a>(&'a self) -> impl Foo<Associated=()> + 'a {
        X
    }
}

impl Y {
    pub fn returns_unsized<'a>(&'a self) -> Box<impl ?Sized + Foo<Associated=()> + 'a> {
        Box::new(X)
    }
}
