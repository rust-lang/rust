// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-test
// This checks that preemption works.

// note: halfway done porting to modern rust
extern mod extra;

use std::comm;
use extra::comm;

fn starve_main(alive: Port<int>) {
    debug!("signalling main");
    alive.recv();
    debug!("starving main");
    let mut i: int = 0;
    loop { i += 1; }
}

pub fn main() {
    let (port, chan) = stream();

    debug!("main started");
    do spawn {
        starve_main(port);
    };
    let mut i: int = 0;
    debug!("main waiting for alive signal");
    chan.send(i);
    debug!("main got alive signal");
    while i < 50 { debug!("main iterated"); i += 1; }
    debug!("main completed");
}
