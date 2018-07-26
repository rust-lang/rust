// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(optin_builtin_traits)]

auto trait MySafeTrait {}

struct Foo;

unsafe impl MySafeTrait for Foo {}
//~^ ERROR implementing the trait `MySafeTrait` is not unsafe

unsafe auto trait MyUnsafeTrait {}

impl MyUnsafeTrait for Foo {}
//~^ ERROR the trait `MyUnsafeTrait` requires an `unsafe impl` declaration

fn main() {}
