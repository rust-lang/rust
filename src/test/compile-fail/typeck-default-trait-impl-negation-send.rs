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

struct MySendable {
   t: *mut u8
}

unsafe impl Send for MySendable {}

struct MyNotSendable {
   t: *mut u8
}

impl !Send for MyNotSendable {}

fn is_send<T: Send>() {}

fn main() {
    is_send::<MySendable>();
    is_send::<MyNotSendable>();
    //~^ ERROR `MyNotSendable` cannot be sent between threads safely
}
