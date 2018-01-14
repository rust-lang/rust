// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(optin_builtin_traits)]

auto trait MyTrait {}

impl<T> !MyTrait for *mut T {}

struct MyS;

struct MyS2;

impl !MyTrait for MyS2 {}

struct MyS3;

fn is_mytrait<T: MyTrait>() {}

fn main() {
    is_mytrait::<MyS>();

    is_mytrait::<MyS2>();
    //~^ ERROR `MyS2: MyTrait` is not satisfied
}
