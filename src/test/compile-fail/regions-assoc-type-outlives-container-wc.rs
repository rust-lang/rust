// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that we are imposing the requirement that every associated
// type of a bound that appears in the where clause on a struct must
// outlive the location in which the type appears, even when the
// constraint is in a where clause not a bound. Issue #22246.

#![allow(dead_code)]

use std::marker::PhantomFn;

///////////////////////////////////////////////////////////////////////////

pub trait TheTrait: PhantomFn<Self, Self> {
    type TheAssocType;
}

pub struct TheType<'b> {
    m: [fn(&'b()); 0]
}

impl<'b> TheTrait for TheType<'b> {
    type TheAssocType = &'b ();
}

///////////////////////////////////////////////////////////////////////////

pub struct WithAssoc<T> where T : TheTrait {
    m: [T; 0]
}

fn with_assoc<'a,'b>() {
    // For this type to be valid, the rules require that all
    // associated types of traits that appear in `WithAssoc` must
    // outlive 'a. In this case, that means TheType<'b>::TheAssocType,
    // which is &'b (), must outlive 'a.

    let _: &'a WithAssoc<TheType<'b>> = loop { }; //~ ERROR cannot infer
}

fn main() {
}
