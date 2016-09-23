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

#[derive(PartialEq)] //~ ERROR: `derive` may only be applied to structs, enums and unions
trait T { }

#[derive(PartialEq)] //~ ERROR: `derive` may only be applied to structs, enums and unions
impl S { }

#[derive(PartialEq)] //~ ERROR: `derive` may only be applied to structs, enums and unions
impl T for S { }

#[derive(PartialEq)] //~ ERROR: `derive` may only be applied to structs, enums and unions
static s: usize = 0;

#[derive(PartialEq)] //~ ERROR: `derive` may only be applied to structs, enums and unions
const c: usize = 0;

#[derive(PartialEq)] //~ ERROR: `derive` may only be applied to structs, enums and unions
mod m { }

#[derive(PartialEq)] //~ ERROR: `derive` may only be applied to structs, enums and unions
extern "C" { }

#[derive(PartialEq)] //~ ERROR: `derive` may only be applied to structs, enums and unions
type A = usize;

#[derive(PartialEq)] //~ ERROR: `derive` may only be applied to structs, enums and unions
fn main() { }
