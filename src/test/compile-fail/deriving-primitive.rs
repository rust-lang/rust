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
use std::int;

#[deriving(FromPrimitive)]
struct A { x: int }
//~^^ ERROR `FromPrimitive` cannot be derived for structs
//~^^^ ERROR `FromPrimitive` cannot be derived for structs

#[deriving(FromPrimitive)]
struct B(int);
//~^^ ERROR `FromPrimitive` cannot be derived for structs
//~^^^ ERROR `FromPrimitive` cannot be derived for structs

#[deriving(FromPrimitive)]
enum C { Foo(int), Bar(uint) }
//~^^ ERROR `FromPrimitive` cannot be derived for enum variants with arguments
//~^^^ ERROR `FromPrimitive` cannot be derived for enum variants with arguments

#[deriving(FromPrimitive)]
enum D { Baz { x: int } }
//~^^ ERROR `FromPrimitive` cannot be derived for enums with struct variants
//~^^^ ERROR `FromPrimitive` cannot be derived for enums with struct variants

pub fn main() {}
