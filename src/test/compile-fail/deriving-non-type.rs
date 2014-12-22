// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(dead_code)]

struct S;

#[deriving(PartialEq)] //~ ERROR: `deriving` may only be applied to structs and enums
trait T { }

#[deriving(PartialEq)] //~ ERROR: `deriving` may only be applied to structs and enums
impl S { }

#[deriving(PartialEq)] //~ ERROR: `deriving` may only be applied to structs and enums
impl T for S { }

#[deriving(PartialEq)] //~ ERROR: `deriving` may only be applied to structs and enums
static s: uint = 0u;

#[deriving(PartialEq)] //~ ERROR: `deriving` may only be applied to structs and enums
const c: uint = 0u;

#[deriving(PartialEq)] //~ ERROR: `deriving` may only be applied to structs and enums
mod m { }

#[deriving(PartialEq)] //~ ERROR: `deriving` may only be applied to structs and enums
extern "C" { }

#[deriving(PartialEq)] //~ ERROR: `deriving` may only be applied to structs and enums
type A = uint;

#[deriving(PartialEq)] //~ ERROR: `deriving` may only be applied to structs and enums
fn main() { }
