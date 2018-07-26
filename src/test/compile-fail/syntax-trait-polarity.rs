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

use std::marker::Send;

struct TestType;

impl !TestType {}
//~^ ERROR inherent impls cannot be negative

trait TestTrait {}

unsafe impl !Send for TestType {}
//~^ ERROR negative impls cannot be unsafe
impl !TestTrait for TestType {}
//~^ ERROR negative impls are only allowed for auto traits

struct TestType2<T>(T);

impl<T> !TestType2<T> {}
//~^ ERROR inherent impls cannot be negative

unsafe impl<T> !Send for TestType2<T> {}
//~^ ERROR negative impls cannot be unsafe
impl<T> !TestTrait for TestType2<T> {}
//~^ ERROR negative impls are only allowed for auto traits

fn main() {}
