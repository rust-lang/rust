// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-tidy-linelength

use std::cell::Cell;
use std::rc::Rc;

fn send<T: Send>(_: T) {}

fn main() {
}

// Cycles should work as the deferred obligations are
// independently resolved and only require the concrete
// return type, which can't depend on the obligation.
fn cycle1() -> impl Clone {
    //~^ ERROR cycle detected
    //~| ERROR cycle detected
    send(cycle2().clone());
    //~^ ERROR `std::rc::Rc<std::string::String>` cannot be sent between threads safely

    Rc::new(Cell::new(5))
}

fn cycle2() -> impl Clone {
    send(cycle1().clone());

    Rc::new(String::from("foo"))
}
