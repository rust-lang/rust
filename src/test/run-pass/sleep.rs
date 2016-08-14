// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-emscripten no threads support

use std::thread::{self, sleep};
use std::time::Duration;
use std::sync::{Arc, Mutex};
use std::u64;

fn main() {
    let finished = Arc::new(Mutex::new(false));
    let t_finished = finished.clone();
    thread::spawn(move || {
        sleep(Duration::new(u64::MAX, 0));
        *t_finished.lock().unwrap() = true;
    });
    sleep(Duration::from_millis(100));
    assert_eq!(*finished.lock().unwrap(), false);
}
