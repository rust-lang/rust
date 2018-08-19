// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(nll)]
#![allow(unused_variables)]

pub trait TryTransform {
    fn try_transform<F>(self, f: F)
    where
        Self: Sized,
        F: FnOnce(Self);
}

impl<'a, T> TryTransform for &'a mut T {
    fn try_transform<F>(self, f: F)
    where
        // The bug was that `Self: Sized` caused the lifetime of `this` to "extend" for all
        // of 'a instead of only lasting as long as the binding is used (for just that line).
        Self: Sized,
        F: FnOnce(Self),
    {
        let this: *mut T = self as *mut T;
        f(self);
    }
}

fn main() {
}
