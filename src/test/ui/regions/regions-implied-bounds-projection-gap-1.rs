// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-compare-mode-nll

// Illustrates the "projection gap": in this test, even though we know
// that `T::Foo: 'x`, that does not tell us that `T: 'x`, because
// there might be other ways for the caller of `func` to show that
// `T::Foo: 'x` holds (e.g., where-clause).

trait Trait1<'x> {
    type Foo;
}

// calling this fn should trigger a check that the type argument
// supplied is well-formed.
fn wf<T>() { }

fn func<'x, T:Trait1<'x>>(t: &'x T::Foo)
{
    wf::<&'x T>();
    //~^ ERROR the parameter type `T` may not live long enough
}

fn caller2<'x, T:Trait1<'x>>(t: &'x T)
{
    wf::<&'x T::Foo>(); // OK
}

fn caller3<'x, T:Trait1<'x>>(t: &'x T::Foo)
{
    wf::<&'x T::Foo>(); // OK
}

fn main() { }
