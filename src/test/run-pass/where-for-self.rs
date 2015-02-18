// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that we can quantify lifetimes outside a constraint (i.e., including
// the self type) in a where clause.

use std::marker::PhantomFn;

static mut COUNT: u32 = 1;

trait Bar<'a>
    : PhantomFn<&'a ()>
{
    fn bar(&self);
}

trait Baz<'a>
    : PhantomFn<&'a ()>
{
    fn baz(&self);
}

impl<'a, 'b> Bar<'b> for &'a u32 {
    fn bar(&self) {
        unsafe { COUNT *= 2; }
    }
}

impl<'a, 'b> Baz<'b> for &'a u32 {
    fn baz(&self) {
        unsafe { COUNT *= 3; }
    }
}

// Test we can use the syntax for HRL including the self type.
fn foo1<T>(x: &T)
    where for<'a, 'b> &'a T: Bar<'b>
{
    x.bar()
}

// Test we can quantify multiple bounds (i.e., the precedence is sensible).
fn foo2<T>(x: &T)
    where for<'a, 'b> &'a T: Bar<'b> + Baz<'b>
{
    x.baz();
    x.bar()
}

fn main() {
    let x = 42u32;
    foo1(&x);
    foo2(&x);
    unsafe {
        assert!(COUNT == 12);
    }
}

