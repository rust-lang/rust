// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Check that `Self` appearing in a phantom fn does not make a trait not object safe.

#![feature(rustc_attrs)]
#![allow(dead_code)]

use std::marker::PhantomFn;

trait Baz : PhantomFn<Self> {
}

trait Bar<T> : PhantomFn<(Self, T)> {
}

fn make_bar<T:Bar<u32>>(t: &T) -> &Bar<u32> {
    t
}

fn make_baz<T:Baz>(t: &T) -> &Baz {
    t
}

#[rustc_error]
fn main() { //~ ERROR compilation successful
}
