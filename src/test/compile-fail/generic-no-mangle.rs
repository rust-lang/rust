// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![deny(no_mangle_generic_items)]

#[no_mangle]
pub fn foo<T>() {} //~ ERROR functions generic over types must be mangled

#[no_mangle]
pub extern fn bar<T>() {} //~ ERROR functions generic over types must be mangled

#[no_mangle]
pub fn baz(x: &i32) -> &i32 { x }

#[no_mangle]
pub fn qux<'a>(x: &'a i32) -> &i32 { x }

fn main() {}
