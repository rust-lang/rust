// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-tidy-linelength

#![feature(infer_outlives_requirements)]

/*
 * We don't infer `T: 'static` outlives relationships by default.
 * Instead an additional feature gate `infer_static_outlives_requirements`
 * is required.
 */

struct Foo<U> {
    bar: Bar<U> //~ ERROR 22:5: 22:16: the parameter type `U` may not live long enough [E0310]
}
struct Bar<T: 'static> {
    x: T,
}

fn main() {}

