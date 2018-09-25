// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test for RFC 1268: we allow overlapping impls of marker traits,
// that is, traits with #[marker]. In this case, a type `T` is
// `MyMarker` if it is either `Debug` or `Display`. This test just
// checks that we don't consider **all** types to be `MyMarker`.

#![feature(marker_trait_attr)]

use std::fmt::{Debug, Display};

#[marker] trait Marker {}

impl<T: Debug> Marker for T {}
impl<T: Display> Marker for T {}

fn is_marker<T: Marker>() { }

struct NotDebugOrDisplay;

fn main() {
    // Debug && Display:
    is_marker::<i32>();

    // Debug && !Display:
    is_marker::<Vec<i32>>();

    // !Debug && !Display
    is_marker::<NotDebugOrDisplay>(); //~ ERROR
}
