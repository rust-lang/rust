// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn is_send<T: Send>() {}
fn is_freeze<T: Share>() {}
fn is_static<T: 'static>() {}

fn main() {
    is_send::<proc()>();
    //~^ ERROR: instantiating a type parameter with an incompatible type

    is_freeze::<proc()>();
    //~^ ERROR: instantiating a type parameter with an incompatible type

    is_static::<proc()>();
    //~^ ERROR: instantiating a type parameter with an incompatible type
}

