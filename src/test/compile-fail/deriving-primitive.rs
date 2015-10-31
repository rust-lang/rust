// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::num::FromPrimitive;
use std::isize;

#[derive(FromPrimitive)]
struct A { x: isize }
//~^^ ERROR `FromPrimitive` cannot be derived for structs
//~^^^ ERROR `FromPrimitive` cannot be derived for structs

#[derive(FromPrimitive)]
struct B(isize);
//~^^ ERROR `FromPrimitive` cannot be derived for structs
//~^^^ ERROR `FromPrimitive` cannot be derived for structs

#[derive(FromPrimitive)]
enum C { Foo(isize), Bar(usize) }
//~^^ ERROR `FromPrimitive` cannot be derived for enums with non-unit variants
//~^^^ ERROR `FromPrimitive` cannot be derived for enums with non-unit variants

#[derive(FromPrimitive)]
enum D { Baz { x: isize } }
//~^^ ERROR `FromPrimitive` cannot be derived for enums with non-unit variants
//~^^^ ERROR `FromPrimitive` cannot be derived for enums with non-unit variants

pub fn main() {}
