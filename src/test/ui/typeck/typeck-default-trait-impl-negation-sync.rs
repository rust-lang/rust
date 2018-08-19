// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-tidy-linelength

#![feature(optin_builtin_traits)]

struct Managed;
impl !Send for Managed {}
impl !Sync for Managed {}

use std::cell::UnsafeCell;

struct MySync {
   t: *mut u8
}

unsafe impl Sync for MySync {}

struct MyNotSync {
   t: *mut u8
}

impl !Sync for MyNotSync {}

struct MyTypeWUnsafe {
   t: UnsafeCell<u8>
}

struct MyTypeManaged {
   t: Managed
}

fn is_sync<T: Sync>() {}

fn main() {
    is_sync::<MySync>();
    is_sync::<MyNotSync>();
    //~^ ERROR `MyNotSync` cannot be shared between threads safely [E0277]

    is_sync::<MyTypeWUnsafe>();
    //~^ ERROR `std::cell::UnsafeCell<u8>` cannot be shared between threads safely [E0277]

    is_sync::<MyTypeManaged>();
    //~^ ERROR `Managed` cannot be shared between threads safely [E0277]
}
