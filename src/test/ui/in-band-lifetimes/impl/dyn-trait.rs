// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that `impl MyTrait<'_> for &i32` is equivalent to `impl<'a,
// 'b> MyTrait<'a> for &'b i32`.

#![allow(warnings)]

#![feature(in_band_lifetimes)]

use std::fmt::Debug;

// Equivalent to `Box<dyn Debug + 'static>`:
trait StaticTrait { }
impl StaticTrait for Box<dyn Debug> { }

// Equivalent to `Box<dyn Debug + 'static>`:
trait NotStaticTrait { }
impl NotStaticTrait for Box<dyn Debug + '_> { }

fn static_val<T: StaticTrait>(_: T) {
}

fn with_dyn_debug_static<'a>(x: Box<dyn Debug + 'a>) {
    static_val(x); //~ ERROR cannot infer
}

fn not_static_val<T: NotStaticTrait>(_: T) {
}

fn with_dyn_debug_not_static<'a>(x: Box<dyn Debug + 'a>) {
    not_static_val(x); // OK
}

fn main() {
}
