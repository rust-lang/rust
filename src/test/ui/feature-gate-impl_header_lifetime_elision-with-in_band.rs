// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(warnings)]

// Make sure this related feature didn't accidentally enable this
#![feature(in_band_lifetimes)]

trait MyTrait<'a> { }

impl MyTrait<'a> for &u32 { }
//~^ ERROR missing lifetime specifier

struct MyStruct;
trait MarkerTrait {}

impl MarkerTrait for &'_ MyStruct { }
//~^ ERROR missing lifetime specifier

fn main() {}
