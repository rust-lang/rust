// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that the variance computation considers types/regions that
// appear in projections to be invariant.

#![feature(rustc_attrs)]

trait Trait<'a> {
    type Type;

    fn method(&'a self) { }
}

#[rustc_variance]
struct Foo<'a, T : Trait<'a>> { //~ ERROR [-, +]
    field: (T, &'a ())
}

#[rustc_variance]
struct Bar<'a, T : Trait<'a>> { //~ ERROR [o, o]
    field: <T as Trait<'a>>::Type
}

fn main() { }
