// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Tests for RFC 1268: we allow overlapping impls of marker traits,
// that is, traits without items. In this case, a type `T` is
// `MyMarker` if it is either `Debug` or `Display`.

#![feature(overlapping_marker_traits)]
#![feature(optin_builtin_traits)]

use std::fmt::{Debug, Display};

trait MyMarker {}

impl<T: Debug> MyMarker for T {}
impl<T: Display> MyMarker for T {}

fn foo<T: MyMarker>(t: T) -> T {
    t
}

fn main() {
    // Debug && Display:
    assert_eq!(1, foo(1));
    assert_eq!(2.0, foo(2.0));

    // Debug && !Display:
    assert_eq!(vec![1], foo(vec![1]));
}
