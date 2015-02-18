// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![deny(bivariance)]
#![allow(dead_code)]
#![feature(rustc_attrs)]

use std::cell::Cell;

// Check that a type parameter which is only used in a trait bound is
// not considered bivariant.

#[rustc_variance]
struct InvariantMut<'a,A:'a,B:'a> { //~ ERROR types=[[o, o];[];[]], regions=[[-];[];[]]
    t: &'a mut (A,B)
}

#[rustc_variance]
struct InvariantCell<A> { //~ ERROR types=[[o];[];[]]
    t: Cell<A>
}

#[rustc_variance]
struct InvariantIndirect<A> { //~ ERROR types=[[o];[];[]]
    t: InvariantCell<A>
}

#[rustc_variance]
struct Covariant<A> { //~ ERROR types=[[+];[];[]]
    t: A, u: fn() -> A
}

#[rustc_variance]
struct Contravariant<A> { //~ ERROR types=[[-];[];[]]
    t: fn(A)
}

#[rustc_variance]
enum Enum<A,B,C> { //~ ERROR types=[[+, -, o];[];[]]
    Foo(Covariant<A>),
    Bar(Contravariant<B>),
    Zed(Covariant<C>,Contravariant<C>)
}

pub fn main() { }
