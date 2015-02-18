// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


extern crate core;

use core::nonzero::NonZero;
use std::mem::size_of;
use std::rc::Rc;
use std::sync::Arc;

trait Trait { fn dummy(&self) { } }

fn main() {
    // Functions
    assert_eq!(size_of::<fn(int)>(), size_of::<Option<fn(int)>>());
    assert_eq!(size_of::<extern "C" fn(int)>(), size_of::<Option<extern "C" fn(int)>>());

    // Slices - &str / &[T] / &mut [T]
    assert_eq!(size_of::<&str>(), size_of::<Option<&str>>());
    assert_eq!(size_of::<&[int]>(), size_of::<Option<&[int]>>());
    assert_eq!(size_of::<&mut [int]>(), size_of::<Option<&mut [int]>>());

    // Traits - Box<Trait> / &Trait / &mut Trait
    assert_eq!(size_of::<Box<Trait>>(), size_of::<Option<Box<Trait>>>());
    assert_eq!(size_of::<&Trait>(), size_of::<Option<&Trait>>());
    assert_eq!(size_of::<&mut Trait>(), size_of::<Option<&mut Trait>>());

    // Pointers - Box<T>
    assert_eq!(size_of::<Box<int>>(), size_of::<Option<Box<int>>>());

    // The optimization can't apply to raw pointers
    assert!(size_of::<Option<*const int>>() != size_of::<*const int>());
    assert!(Some(0 as *const int).is_some()); // Can't collapse None to null

    struct Foo {
        _a: Box<int>
    }
    struct Bar(Box<int>);

    // Should apply through structs
    assert_eq!(size_of::<Foo>(), size_of::<Option<Foo>>());
    assert_eq!(size_of::<Bar>(), size_of::<Option<Bar>>());
    // and tuples
    assert_eq!(size_of::<(u8, Box<int>)>(), size_of::<Option<(u8, Box<int>)>>());
    // and fixed-size arrays
    assert_eq!(size_of::<[Box<int>; 1]>(), size_of::<Option<[Box<int>; 1]>>());

    // Should apply to NonZero
    assert_eq!(size_of::<NonZero<uint>>(), size_of::<Option<NonZero<uint>>>());
    assert_eq!(size_of::<NonZero<*mut i8>>(), size_of::<Option<NonZero<*mut i8>>>());

    // Should apply to types that use NonZero internally
    assert_eq!(size_of::<Vec<int>>(), size_of::<Option<Vec<int>>>());
    assert_eq!(size_of::<Arc<int>>(), size_of::<Option<Arc<int>>>());
    assert_eq!(size_of::<Rc<int>>(), size_of::<Option<Rc<int>>>());

    // Should apply to types that have NonZero transitively
    assert_eq!(size_of::<String>(), size_of::<Option<String>>());

}
