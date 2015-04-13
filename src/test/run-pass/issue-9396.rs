// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// pretty-expanded FIXME #23616

#![feature(old_io, std_misc)]

use std::sync::mpsc::{TryRecvError, channel};
use std::old_io::timer::Timer;
use std::thread;
use std::time::Duration;

pub fn main() {
    let (tx, rx) = channel();
    let t = thread::spawn(move||{
        let mut timer = Timer::new().unwrap();
        timer.sleep(Duration::milliseconds(10));
        tx.send(()).unwrap();
    });
    loop {
        match rx.try_recv() {
            Ok(()) => break,
            Err(TryRecvError::Empty) => {}
            Err(TryRecvError::Disconnected) => unreachable!()
        }
    }
    t.join();
}
