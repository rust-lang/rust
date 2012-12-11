// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn of<T>() -> @fn(T) { fail; }
fn subtype<T>(x: @fn(T)) { fail; }

fn test_fn<T>(_x: &x/T, _y: &y/T, _z: &z/T) {
    // Here, x, y, and z are free.  Other letters
    // are bound.  Note that the arrangement
    // subtype::<T1>(of::<T2>()) will typecheck
    // iff T1 <: T2.

    subtype::<fn(&a/T)>(
        of::<fn(&a/T)>());

    subtype::<fn(&a/T)>(
        of::<fn(&b/T)>());

    subtype::<fn(&b/T)>(
        of::<fn(&x/T)>());

    subtype::<fn(&x/T)>(
        of::<fn(&b/T)>());  //~ ERROR mismatched types

    subtype::<fn(&a/T, &b/T)>(
        of::<fn(&a/T, &a/T)>());

    subtype::<fn(&a/T, &a/T)>(
        of::<fn(&a/T, &b/T)>()); //~ ERROR mismatched types

    subtype::<fn(&a/T, &b/T)>(
        of::<fn(&x/T, &y/T)>());

    subtype::<fn(&x/T, &y/T)>(
        of::<fn(&a/T, &b/T)>()); //~ ERROR mismatched types

    subtype::<fn(&x/T) -> @fn(&a/T)>(
        of::<fn(&x/T) -> @fn(&a/T)>());

    subtype::<fn(&a/T) -> @fn(&a/T)>(
        of::<fn(&a/T) -> @fn(&b/T)>()); //~ ERROR mismatched types

    subtype::<fn(&a/T) -> @fn(&a/T)>(
        of::<fn(&x/T) -> @fn(&b/T)>()); //~ ERROR mismatched types

    subtype::<fn(&a/T) -> @fn(&b/T)>(
        of::<fn(&a/T) -> @fn(&a/T)>());
}

fn main() {}