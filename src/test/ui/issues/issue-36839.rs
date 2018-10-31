// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-pass
// skip-codegen

pub trait Foo {
    type Bar;
}

pub trait Broken {
    type Assoc;
    fn broken(&self) where Self::Assoc: Foo;
}

impl<T> Broken for T {
    type Assoc = ();
    fn broken(&self) where Self::Assoc: Foo {
        let _x: <Self::Assoc as Foo>::Bar;
    }
}


fn main() {
    let _m: &Broken<Assoc=()> = &();
}
