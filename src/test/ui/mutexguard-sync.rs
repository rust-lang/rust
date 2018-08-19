// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// MutexGuard<Cell<i32>> must not be Sync, that would be unsound.
use std::sync::Mutex;
use std::cell::Cell;

fn test_sync<T: Sync>(_t: T) {}

fn main()
{
    let m = Mutex::new(Cell::new(0i32));
    let guard = m.lock().unwrap();
    test_sync(guard);
    //~^ ERROR `std::cell::Cell<i32>` cannot be shared between threads safely [E0277]
}
