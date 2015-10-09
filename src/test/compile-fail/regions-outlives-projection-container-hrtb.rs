// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that structs with higher-ranked where clauses don't generate
// "outlives" requirements. Issue #22246.

#![allow(dead_code)]
#![feature(rustc_attrs)]

///////////////////////////////////////////////////////////////////////////

pub trait TheTrait<'b> {
    type TheAssocType;
}

pub struct TheType<'b> {
    m: [fn(&'b()); 0]
}

impl<'a,'b> TheTrait<'a> for TheType<'b> {
    type TheAssocType = &'b ();
}

///////////////////////////////////////////////////////////////////////////

pub struct WithHrAssoc<T>
    where for<'a> T : TheTrait<'a>
{
    m: [T; 0]
}

fn with_assoc<'a,'b>() {
    // We get an error because 'b:'a does not hold:

    let _: &'a WithHrAssoc<TheType<'b>> = loop { };
    //~^ ERROR reference has a longer lifetime
}

///////////////////////////////////////////////////////////////////////////

pub trait TheSubTrait : for<'a> TheTrait<'a> {
}

impl<'b> TheSubTrait for TheType<'b> { }

pub struct WithHrAssocSub<T>
    where T : TheSubTrait
{
    m: [T; 0]
}

fn with_assoc_sub<'a,'b>() {
    // The error here is just because `'b:'a` must hold for the type
    // below to be well-formed, it is not related to the HR relation.

    let _: &'a WithHrAssocSub<TheType<'b>> = loop { };
    //~^ ERROR reference has a longer lifetime
}

#[rustc_error]
fn main() {
}
