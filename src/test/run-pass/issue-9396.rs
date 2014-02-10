// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::comm;
use std::io::timer::Timer;

pub fn main() {
    let (port, chan) = Chan::new();
    spawn(proc (){
        let mut timer = Timer::new().unwrap();
        timer.sleep(10);
        chan.send(());
    });
    loop {
        match port.try_recv() {
            comm::Data(()) => break,
            comm::Empty => {}
            comm::Disconnected => unreachable!()
        }
    }
}
