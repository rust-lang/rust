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

use pipes::{Select2, Selectable};

fn main() {
    let (p,c) = pipes::stream();
    do task::try |move c| {
        let (p2,c2) = pipes::stream();
        do task::spawn |move p2| {
            p2.recv();
            error!("sibling fails");
            fail;
        }   
        let (p3,c3) = pipes::stream();
        c.send(move c3);
        c2.send(());
        error!("child blocks");
        let (p, c) = pipes::stream();
        (move p, move p3).select();
        c.send(());
    };  
    error!("parent tries");
    assert !p.recv().try_send(());
    error!("all done!");
}
