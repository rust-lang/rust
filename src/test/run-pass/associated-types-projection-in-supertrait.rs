// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that we are handle to correctly handle a projection type
// that appears in a supertrait bound. Issue #20559.

trait A
{
    type TA;

    fn dummy(&self) { }
}

trait B<TB>
{
    fn foo (&self, t : TB) -> String;
}

trait C<TC : A> : B<<TC as A>::TA> { }

struct X;

impl A for X
{
    type TA = i32;
}

struct Y;

impl C<X> for Y { }

// Both of these impls are required for successful compilation
impl B<i32> for Y
{
    fn foo (&self, t : i32) -> String
    {
        format!("First {}", t)
    }
}

fn main ()
{
    let y = Y;
    assert_eq!(y.foo(5), format!("First 5"));
}
