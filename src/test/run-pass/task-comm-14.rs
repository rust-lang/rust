// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-fast

use std::comm;
use std::task;

pub fn main() {
    let po = comm::PortSet::new();

    // Spawn 10 tasks each sending us back one int.
    let mut i = 10;
    while (i > 0) {
        debug!(i);
        let (p, ch) = comm::stream();
        po.add(p);
        task::spawn({let i = i; || child(i, &ch)});
        i = i - 1;
    }

    // Spawned tasks are likely killed before they get a chance to send
    // anything back, so we deadlock here.

    i = 10;
    while (i > 0) {
        debug!(i);
        po.recv();
        i = i - 1;
    }

    debug!("main thread exiting");
}

fn child(x: int, ch: &comm::Chan<int>) {
    debug!(x);
    ch.send(x);
}
