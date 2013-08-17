// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[allow(dead_assignment)];

use std::comm::SharedChan;
use std::comm;

pub fn main() { test00(); }

fn test00() {
    let mut r: int = 0;
    let mut sum: int = 0;
    let (p, ch) = comm::stream();
    let ch = SharedChan::new(ch);
    let c0 = ch.clone();
    let c1 = ch.clone();
    let c2 = ch.clone();
    let c3 = ch.clone();
    let number_of_messages: int = 1000;
    let mut i: int = 0;
    while i < number_of_messages {
        c0.send(i + 0);
        c1.send(i + 0);
        c2.send(i + 0);
        c3.send(i + 0);
        i += 1;
    }
    i = 0;
    while i < number_of_messages {
        r = p.recv();
        sum += r;
        r = p.recv();
        sum += r;
        r = p.recv();
        sum += r;
        r = p.recv();
        sum += r;
        i += 1;
    }
    assert_eq!(sum, 1998000);
    // assert (sum == 4 * ((number_of_messages *
    //                   (number_of_messages - 1)) / 2));

}
