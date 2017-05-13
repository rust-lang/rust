// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(optin_builtin_traits)]

trait MyTrait {}

struct TestType<T>(::std::marker::PhantomData<T>);

unsafe impl<T: MyTrait+'static> Send for TestType<T> {}

impl<T: MyTrait> !Send for TestType<T> {}
//~^ ERROR conflicting implementations of trait `std::marker::Send`

unsafe impl<T:'static> Send for TestType<T> {}
//~^ ERROR conflicting implementations of trait `std::marker::Send`

impl !Send for TestType<i32> {}

fn main() {}
