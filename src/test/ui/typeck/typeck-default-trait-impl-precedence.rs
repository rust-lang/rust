// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that declaring that `&T` is `Defaulted` if `T:Signed` implies
// that other `&T` is NOT `Defaulted` if `T:Signed` does not hold. In
// other words, the auto impl only applies if there are no existing
// impls whose types unify.

#![feature(optin_builtin_traits)]

auto trait Defaulted { }
impl<'a,T:Signed> Defaulted for &'a T { }
impl<'a,T:Signed> Defaulted for &'a mut T { }
fn is_defaulted<T:Defaulted>() { }

trait Signed { }
impl Signed for i32 { }

fn main() {
    is_defaulted::<&'static i32>();
    is_defaulted::<&'static u32>();
    //~^ ERROR `u32: Signed` is not satisfied
}
