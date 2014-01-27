// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-test linked failure
// xfail-fast

use std::comm;
use std::task;

pub fn main() {
    let (p,c) = comm::stream();
    task::try(|| {
        let (p2,c2) = comm::stream();
        task::spawn(|| {
            p2.recv();
            error!("sibling fails");
            fail!();
        });
        let (p3,c3) = comm::stream();
        c.send(c3);
        c2.send(());
        error!("child blocks");
        p3.recv();
    });
    error!("parent tries");
    assert!(!p.recv().try_send(()));
    error!("all done!");
}
