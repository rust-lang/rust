// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// pretty-expanded FIXME #23616

#![feature(core)]

extern crate core;

use core::nonzero::NonZero;
use std::mem::size_of;
use std::rc::Rc;
use std::sync::Arc;

trait Trait { fn dummy(&self) { } }

fn main() {
    // Functions
    assert_eq!(size_of::<fn(isize)>(), size_of::<Option<fn(isize)>>());
    assert_eq!(size_of::<extern "C" fn(isize)>(), size_of::<Option<extern "C" fn(isize)>>());

    // Slices - &str / &[T] / &mut [T]
    assert_eq!(size_of::<&str>(), size_of::<Option<&str>>());
    assert_eq!(size_of::<&[isize]>(), size_of::<Option<&[isize]>>());
    assert_eq!(size_of::<&mut [isize]>(), size_of::<Option<&mut [isize]>>());

    // Traits - Box<Trait> / &Trait / &mut Trait
    assert_eq!(size_of::<Box<Trait>>(), size_of::<Option<Box<Trait>>>());
    assert_eq!(size_of::<&Trait>(), size_of::<Option<&Trait>>());
    assert_eq!(size_of::<&mut Trait>(), size_of::<Option<&mut Trait>>());

    // Pointers - Box<T>
    assert_eq!(size_of::<Box<isize>>(), size_of::<Option<Box<isize>>>());

    // The optimization can't apply to raw pointers
    assert!(size_of::<Option<*const isize>>() != size_of::<*const isize>());
    assert!(Some(0 as *const isize).is_some()); // Can't collapse None to null

    struct Foo {
        _a: Box<isize>
    }
    struct Bar(Box<isize>);

    // Should apply through structs
    assert_eq!(size_of::<Foo>(), size_of::<Option<Foo>>());
    assert_eq!(size_of::<Bar>(), size_of::<Option<Bar>>());
    // and tuples
    assert_eq!(size_of::<(u8, Box<isize>)>(), size_of::<Option<(u8, Box<isize>)>>());
    // and fixed-size arrays
    assert_eq!(size_of::<[Box<isize>; 1]>(), size_of::<Option<[Box<isize>; 1]>>());

    // Should apply to NonZero
    assert_eq!(size_of::<NonZero<usize>>(), size_of::<Option<NonZero<usize>>>());
    assert_eq!(size_of::<NonZero<*mut i8>>(), size_of::<Option<NonZero<*mut i8>>>());

    // Should apply to types that use NonZero internally
    assert_eq!(size_of::<Vec<isize>>(), size_of::<Option<Vec<isize>>>());
    assert_eq!(size_of::<Arc<isize>>(), size_of::<Option<Arc<isize>>>());
    assert_eq!(size_of::<Rc<isize>>(), size_of::<Option<Rc<isize>>>());

    // Should apply to types that have NonZero transitively
    assert_eq!(size_of::<String>(), size_of::<Option<String>>());

}
