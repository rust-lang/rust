// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(marker_trait_attr)]

#[marker] //~ ERROR attribute can only be applied to a trait
struct Struct {}

#[marker] //~ ERROR attribute can only be applied to a trait
impl Struct {}

#[marker] //~ ERROR attribute can only be applied to a trait
union Union {
    x: i32,
}

#[marker] //~ ERROR attribute can only be applied to a trait
const CONST: usize = 10;

#[marker] //~ ERROR attribute can only be applied to a trait
fn function() {}

#[marker] //~ ERROR attribute can only be applied to a trait
type Type = ();

fn main() {}
