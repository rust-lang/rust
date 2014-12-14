// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test what happens when a HR obligation is applie to an impl with
// "outlives" bounds. Currently we're pretty conservative here; this
// will probably improve in time.

trait Foo<X> {
    fn foo(&self, x: X) { }
}

fn want_foo<T>()
    where T : for<'a> Foo<&'a int>
{
}

///////////////////////////////////////////////////////////////////////////
// Expressed as a where clause

struct SomeStruct<X> {
    x: X
}

impl<'a,X> Foo<&'a int> for SomeStruct<X>
    where X : 'a
{
}

fn one() {
    // In fact there is no good reason for this to be an error, but
    // whatever, I'm mostly concerned it doesn't ICE right now:
    want_foo::<SomeStruct<uint>>();
    //~^ ERROR requirement `for<'a> uint : 'a` is not satisfied
}

///////////////////////////////////////////////////////////////////////////
// Expressed as shorthand

struct AnotherStruct<X> {
    x: X
}

impl<'a,X:'a> Foo<&'a int> for AnotherStruct<X>
{
}

fn two() {
    want_foo::<AnotherStruct<uint>>();
    //~^ ERROR requirement `for<'a> uint : 'a` is not satisfied
}

fn main() { }
