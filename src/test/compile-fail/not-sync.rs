// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::cell::{Cell, RefCell};
use std::rc::{Rc, Weak};
use std::sync::mpsc::{Receiver, Sender};

fn test<T: Sync>() {}

fn main() {
    test::<Cell<i32>>();
    //~^ ERROR `std::cell::Cell<i32>` cannot be shared between threads safely [E0277]
    test::<RefCell<i32>>();
    //~^ ERROR `std::cell::RefCell<i32>` cannot be shared between threads safely [E0277]

    test::<Rc<i32>>();
    //~^ ERROR `std::rc::Rc<i32>` cannot be shared between threads safely [E0277]
    test::<Weak<i32>>();
    //~^ ERROR `std::rc::Weak<i32>` cannot be shared between threads safely [E0277]

    test::<Receiver<i32>>();
    //~^ ERROR `std::sync::mpsc::Receiver<i32>` cannot be shared between threads safely [E0277]
    test::<Sender<i32>>();
    //~^ ERROR `std::sync::mpsc::Sender<i32>` cannot be shared between threads safely [E0277]
}
