// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -Z parse-only
// must-compile-successfully

#![feature(generic_associated_types)]

use std::ops::Deref;

trait Foo {
    type Bar<'a>;
    type Bar<'a, 'b>;
    type Bar<'a, 'b,>;
    type Bar<'a, 'b, T>;
    type Bar<'a, 'b, T, U>;
    type Bar<'a, 'b, T, U,>;
    type Bar<'a, 'b, T: Debug, U,>;
    type Bar<'a, 'b, T: Debug, U,>: Debug;
    type Bar<'a, 'b, T: Debug, U,>: Deref<Target = T> + Into<U>;
    type Bar<'a, 'b, T: Debug, U,> where T: Deref<Target = U>, U: Into<T>;
    type Bar<'a, 'b, T: Debug, U,>: Deref<Target = T> + Into<U>
        where T: Deref<Target = U>, U: Into<T>;
}

fn main() {}
