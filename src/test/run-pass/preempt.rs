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

fn starve_main(alive: chan<int>) {
    debug!("signalling main");
    alive.recv(1);
    debug!("starving main");
    let i: int = 0;
    loop { i += 1; }
}

pub fn main() {
    let alive: port<int> = port();
    debug!("main started");
    let s: task = do task::spawn {
        starve_main(chan(alive));
    };
    let i: int;
    debug!("main waiting for alive signal");
    alive.send(i);
    debug!("main got alive signal");
    while i < 50 { debug!("main iterated"); i += 1; }
    debug!("main completed");
}
