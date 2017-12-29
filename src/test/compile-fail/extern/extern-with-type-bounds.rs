// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(intrinsics)]

extern "rust-intrinsic" {
    // Real example from libcore
    fn type_id<T: ?Sized + 'static>() -> u64;

    // Silent bounds made explicit to make sure they are actually
    // resolved.
    fn transmute<T: Sized, U: Sized>(val: T) -> U;

    // Bounds aren't checked right now, so this should work
    // even though it's incorrect.
    fn size_of<T: Clone>() -> usize;

    // Unresolved bounds should still error.
    fn align_of<T: NoSuchTrait>() -> usize;
    //~^ ERROR cannot find trait `NoSuchTrait` in this scope
}

fn main() {}
