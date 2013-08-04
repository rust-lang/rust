// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::comm::*;
use std::task;

fn child(c: &SharedChan<~uint>, i: uint) {
    c.send(~i);
}

pub fn main() {
    let (p, ch) = stream();
    let ch = SharedChan::new(ch);
    let n = 100u;
    let mut expected = 0u;
    for i in range(0u, n) {
        let ch = ch.clone();
        task::spawn(|| child(&ch, i) );
        expected += i;
    }

    let mut actual = 0u;
    for _ in range(0u, n) {
        let j = p.recv();
        actual += *j;
    }

    assert_eq!(expected, actual);
}
