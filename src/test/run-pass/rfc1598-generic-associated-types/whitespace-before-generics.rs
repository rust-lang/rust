// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(generic_associated_types)]

// Checking the interaction with this other feature
#![feature(associated_type_defaults)]

use std::fmt::{Display, Debug};

trait Foo {
    type Assoc where Self: Sized;
    type Assoc2 <T >where T: Display;
    type WithDefault <T> = Iterator <Item=T> where T: Debug;
    // No generics on this associated type
    type NoGenerics;
}

struct Bar;

impl Foo for Bar {
    type Assoc = usize;
    type Assoc2 <T> = Vec<T>;
    type WithDefault<'a, T> = &'a Iterator<T>;
    type NoGenerics = f64;
}

fn main() {}
