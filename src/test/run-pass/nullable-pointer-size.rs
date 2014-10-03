// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(macro_rules)]

use std::mem;

enum E<T> { Thing(int, T), Nothing((), ((), ()), [i8, ..0]) }
struct S<T>(int, T);

// These are macros so we get useful assert messages.

macro_rules! check_option {
    ($T:ty) => {
        assert_eq!(mem::size_of::<Option<$T>>(), mem::size_of::<$T>());
    }
}

macro_rules! check_fancy {
    ($T:ty) => {
        assert_eq!(mem::size_of::<E<$T>>(), mem::size_of::<S<$T>>());
    }
}

macro_rules! check_type {
    ($T:ty) => {{
        check_option!($T);
        check_fancy!($T);
    }}
}

pub fn main() {
    check_type!(&'static int);
    check_type!(Box<int>);
    check_type!(extern fn());
}
