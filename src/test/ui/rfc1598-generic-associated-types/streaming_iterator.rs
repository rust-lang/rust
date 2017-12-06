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

//FIXME(#44265): "lifetime parameter not allowed on this type" errors will be addressed in a
// follow-up PR

use std::fmt::Display;

trait StreamingIterator {
    type Item<'a>;
    // Applying the lifetime parameter `'a` to `Self::Item` inside the trait.
    fn next<'a>(&'a self) -> Option<Self::Item<'a>>;
    //~^ ERROR lifetime parameters are not allowed on this type [E0110]
}

struct Foo<T: StreamingIterator> {
    // Applying a concrete lifetime to the constructor outside the trait.
    bar: <T as StreamingIterator>::Item<'static>,
    //~^ ERROR lifetime parameters are not allowed on this type [E0110]
}

// Users can bound parameters by the type constructed by that trait's associated type constructor
// of a trait using HRTB. Both type equality bounds and trait bounds of this kind are valid:
//FIXME(sunjay): This next line should parse and be valid
//fn foo<T: for<'a> StreamingIterator<Item<'a>=&'a [i32]>>(iter: T) { /* ... */ }
fn foo<T>(iter: T) where T: StreamingIterator, for<'a> T::Item<'a>: Display { /* ... */ }
//~^ ERROR lifetime parameters are not allowed on this type [E0110]

fn main() {}
