// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test a case where you have an impl of `Foo<X>` for all `X` that
// is being applied to `for<'a> Foo<&'a mut X>`. Issue #19730.

trait Foo<X> {
    fn foo(&self, x: X) { }
}

fn want_hrtb<T>()
    where T : for<'a> Foo<&'a int>
{
}

// AnyInt implements Foo<&'a int> for any 'a, so it is a match.
struct AnyInt;
impl<'a> Foo<&'a int> for AnyInt { }
fn give_any() {
    want_hrtb::<AnyInt>()
}

// StaticInt only implements Foo<&'a int> for 'a, so it is an error.
struct StaticInt;
impl Foo<&'static int> for StaticInt { }
fn give_static() {
    want_hrtb::<StaticInt>() //~ ERROR `for<'a> Foo<&'a int>` is not implemented
}

fn main() { }
