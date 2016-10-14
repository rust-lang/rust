// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Check that traits with various kinds of associated items cause
// dropck to inject extra region constraints.

#![allow(non_camel_case_types)]

trait HasSelfMethod { fn m1(&self) { } }
trait HasMethodWithSelfArg { fn m2(x: &Self) { } }
trait HasType { type Something; }

impl HasSelfMethod for i32 { }
impl HasMethodWithSelfArg for i32 { }
impl HasType for i32 { type Something = (); }

impl<'a,T> HasSelfMethod for &'a T { }
impl<'a,T> HasMethodWithSelfArg for &'a T { }
impl<'a,T> HasType for &'a T { type Something = (); }

// e.g. `impl_drop!(Send, D_Send)` expands to:
//   ```rust
//   struct D_Send<T:Send>(T);
//   impl<T:Send> Drop for D_Send<T> { fn drop(&mut self) { } }
//   ```
macro_rules! impl_drop {
    ($Bound:ident, $Id:ident) => {
        struct $Id<T:$Bound>(T);
        impl <T:$Bound> Drop for $Id<T> { fn drop(&mut self) { } }
    }
}

impl_drop!{HasSelfMethod,        D_HasSelfMethod}
impl_drop!{HasMethodWithSelfArg, D_HasMethodWithSelfArg}
impl_drop!{HasType,              D_HasType}

fn f_sm() {
    let (_d, d1);
    d1 = D_HasSelfMethod(1);
    _d = D_HasSelfMethod(&d1);
}
//~^ ERROR `d1` does not live long enough
fn f_mwsa() {
    let (_d, d1);
    d1 = D_HasMethodWithSelfArg(1);
    _d = D_HasMethodWithSelfArg(&d1);
}
//~^ ERROR `d1` does not live long enough
fn f_t() {
    let (_d, d1);
    d1 = D_HasType(1);
    _d = D_HasType(&d1);
}
//~^ ERROR `d1` does not live long enough

fn main() {
    f_sm();
    f_mwsa();
    f_t();
}
