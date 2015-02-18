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

#[derive(PartialEq)] //~ ERROR: `derive` may only be applied to structs and enums
trait T { }

#[derive(PartialEq)] //~ ERROR: `derive` may only be applied to structs and enums
impl S { }

#[derive(PartialEq)] //~ ERROR: `derive` may only be applied to structs and enums
impl T for S { }

#[derive(PartialEq)] //~ ERROR: `derive` may only be applied to structs and enums
static s: usize = 0_usize;

#[derive(PartialEq)] //~ ERROR: `derive` may only be applied to structs and enums
const c: usize = 0_usize;

#[derive(PartialEq)] //~ ERROR: `derive` may only be applied to structs and enums
mod m { }

#[derive(PartialEq)] //~ ERROR: `derive` may only be applied to structs and enums
extern "C" { }

#[derive(PartialEq)] //~ ERROR: `derive` may only be applied to structs and enums
type A = usize;

#[derive(PartialEq)] //~ ERROR: `derive` may only be applied to structs and enums
fn main() { }
