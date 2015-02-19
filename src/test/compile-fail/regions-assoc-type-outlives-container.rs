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
// outlive the location in which the type appears. Issue #22246.

#![allow(dead_code)]
#![feature(rustc_attrs)]

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

pub struct WithAssoc<T:TheTrait> {
    m: [T; 0]
}

pub struct WithoutAssoc<T> {
    m: [T; 0]
}

fn with_assoc<'a,'b>() {
    // For this type to be valid, the rules require that all
    // associated types of traits that appear in `WithAssoc` must
    // outlive 'a. In this case, that means TheType<'b>::TheAssocType,
    // which is &'b (), must outlive 'a.

    let _: &'a WithAssoc<TheType<'b>> = loop { }; //~ ERROR cannot infer
}

fn with_assoc1<'a,'b>() where 'b : 'a {
    // For this type to be valid, the rules require that all
    // associated types of traits that appear in `WithAssoc` must
    // outlive 'a. In this case, that means TheType<'b>::TheAssocType,
    // which is &'b (), must outlive 'a, so 'b : 'a must hold, and
    // that is in the where clauses, so we're fine.

    let _: &'a WithAssoc<TheType<'b>> = loop { };
}

fn without_assoc<'a,'b>() {
    // Here there are no associated types and the `'b` appearing in
    // `TheType<'b>` is purely covariant, so there is no requirement
    // that `'b:'a` holds.

    let _: &'a WithoutAssoc<TheType<'b>> = loop { };
}

fn call_with_assoc<'a,'b>() {
    // As `with_assoc`, but just checking that we impose the same rule
    // on the value supplied for the type argument, even when there is
    // no data.

    call::<&'a WithAssoc<TheType<'b>>>();
    //~^ ERROR cannot infer
}

fn call_without_assoc<'a,'b>() {
    // As `without_assoc`, but in a distinct scenario.

    call::<&'a WithoutAssoc<TheType<'b>>>();
}

fn call<T>() { }

fn main() {
}
