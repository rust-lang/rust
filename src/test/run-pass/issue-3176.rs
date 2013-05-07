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

use core::comm::{Select2, Selectable};

pub fn main() {
    let (p,c) = comm::stream();
    do task::try || {
        let (p2,c2) = comm::stream();
        do task::spawn || {
            p2.recv();
            error!("sibling fails");
            fail!();
        }
        let (p3,c3) = comm::stream();
        c.send(c3);
        c2.send(());
        error!("child blocks");
        let (p, c) = comm::stream();
        let mut tuple = (p, p3);
        tuple.select();
        c.send(());
    };
    error!("parent tries");
    assert!(!p.recv().try_send(()));
    error!("all done!");
}
