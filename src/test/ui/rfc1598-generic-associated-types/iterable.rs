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

//FIXME(#44265): "undeclared lifetime" errors will be addressed in a follow-up PR

trait Iterable {
    type Item<'a>;
    type Iter<'a>: Iterator<Item = Self::Item<'a>>;
    //~^ ERROR undeclared lifetime
    //~| ERROR lifetime parameters are not allowed on this type [E0110]

    fn iter<'a>(&'a self) -> Self::Iter<'a>;
    //~^ ERROR lifetime parameters are not allowed on this type [E0110]
}

fn main() {}
