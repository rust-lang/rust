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
use std::sync::mpsc::{Receiver, Sender, SyncSender};

fn test<T: Sync>() {}

fn main() {
    test::<Cell<i32>>();
    //~^ ERROR marker::Sync` is not implemented for the type `core::cell::Cell<i32>`
    test::<RefCell<i32>>();
    //~^ ERROR marker::Sync` is not implemented for the type `core::cell::RefCell<i32>`

    test::<Rc<i32>>();
    //~^ ERROR marker::Sync` is not implemented for the type `alloc::rc::Rc<i32>`
    test::<Weak<i32>>();
    //~^ ERROR marker::Sync` is not implemented for the type `alloc::rc::Weak<i32>`

    test::<Receiver<i32>>();
    //~^ ERROR marker::Sync` is not implemented for the type `std::sync::mpsc::Receiver<i32>`
    test::<Sender<i32>>();
    //~^ ERROR marker::Sync` is not implemented for the type `std::sync::mpsc::Sender<i32>`
    test::<SyncSender<i32>>();
    //~^ ERROR marker::Sync` is not implemented for the type `std::sync::mpsc::SyncSender<i32>`
}
