// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use std::gc::Gc;
use std::mem::size_of;

trait Trait {}

fn main() {
    // Closures - || / proc()
    assert_eq!(size_of::<proc()>(), size_of::<Option<proc()>>());
    assert_eq!(size_of::<||>(), size_of::<Option<||>>());

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

    // Pointers - Box<T> / Gc<T>
    assert_eq!(size_of::<Box<int>>(), size_of::<Option<Box<int>>>());
    assert_eq!(size_of::<Gc<int>>(), size_of::<Option<Gc<int>>>());


    // The optimization can't apply to raw pointers
    assert!(size_of::<Option<*const int>>() != size_of::<*const int>());
    assert!(Some(0 as *const int).is_some()); // Can't collapse None to null

}
