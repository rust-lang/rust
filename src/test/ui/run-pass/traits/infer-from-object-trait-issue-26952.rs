// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that when we match a trait reference like `Foo<A>: Foo<_#0t>`,
// we unify with `_#0t` with `A`. In this code, if we failed to do
// that, then you get an unconstrained type-variable in `call`.
//
// Also serves as a regression test for issue #26952, though the test
// was derived from another reported regression with the same cause.

use std::marker::PhantomData;

trait Trait<A> { fn foo(&self); }

struct Type<A> { a: PhantomData<A> }

fn as_trait<A>(t: &Type<A>) -> &Trait<A> { loop {  } }

fn want<A,T:Trait<A>+?Sized>(t: &T) { }

fn call<A>(p: Type<A>) {
    let q = as_trait(&p);
    want(q); // parameter A to `want` *would* be unconstrained
}

fn main() { }
