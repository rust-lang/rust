// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -Z parse-only -Z continue-parse-after-error

impl ! {} // OK
impl ! where u8: Copy {} // OK

impl Trait Type {} //~ ERROR missing `for` in a trait impl
impl Trait .. {} //~ ERROR missing `for` in a trait impl
impl ?Sized for Type {} //~ ERROR expected a trait, found type
impl ?Sized for .. {} //~ ERROR expected a trait, found type

default unsafe FAIL //~ ERROR expected `impl`, found `FAIL`
