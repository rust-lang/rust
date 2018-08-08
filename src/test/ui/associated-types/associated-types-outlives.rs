// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Regression test for issue #24622. The older associated types code
// was erroneously assuming that all projections outlived the current
// fn body, causing this (invalid) code to be accepted.

pub trait Foo<'a> {
    type Bar;
}

impl<'a, T:'a> Foo<'a> for T {
    type Bar = &'a T;
}

fn denormalise<'a, T>(t: &'a T) -> <T as Foo<'a>>::Bar {
    t
}

pub fn free_and_use<T: for<'a> Foo<'a>,
                    F: for<'a> FnOnce(<T as Foo<'a>>::Bar)>(x: T, f: F) {
    let y;
    'body: loop { // lifetime annotations added for clarity
        's: loop { y = denormalise(&x); break }
        drop(x); //~ ERROR cannot move out of `x` because it is borrowed
        return f(y);
    }
}

pub fn main() {
}
