// xfail-fast

// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Make sure that we can detect when one end of the pipe is closed.

// xfail-win32

extern mod std;
use std::timer::sleep;
use std::uv;

use core::cell::Cell;
use core::pipes::{try_recv, recv};

proto! oneshot (
    waiting:send {
        signal -> !
    }
)

pub fn main() {
    let iotask = &uv::global_loop::get();
    
    let (chan, port) = oneshot::init();
    let port = Cell(port);
    do spawn {
        match try_recv(port.take()) {
          Some(*) => { fail!() }
          None => { }
        }
    }

    sleep(iotask, 100);

    task::spawn_unlinked(failtest);
}

// Make sure the right thing happens during failure.
fn failtest() {
    let (c, p) = oneshot::init();

    do task::spawn_with(c) |_c| { 
        fail!();
    }

    error!("%?", recv(p));
    // make sure we get killed if we missed it in the receive.
    loop { task::yield() }
}
