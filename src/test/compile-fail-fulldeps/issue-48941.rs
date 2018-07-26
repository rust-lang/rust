// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This is a regression test against an ICE that used to occur
// on malformed attributes for a custom MultiModifier.

// aux-build:macro_crate_test.rs
// ignore-stage1

#![feature(plugin)]
#![plugin(macro_crate_test)]

#[noop_attribute"x"] //~ ERROR expected one of
fn night() { }

#[noop_attribute("hi"), rank = 2] //~ ERROR unexpected token
fn knight() { }

#[noop_attribute("/user", data= = "<user")] //~ ERROR literal or identifier
fn nite() { }

fn main() {}
