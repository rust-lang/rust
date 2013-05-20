// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub fn main() { test00(); }

fn test00() {
    let mut r: int = 0;
    let mut sum: int = 0;
    let (p, c) = comm::stream();
    c.send(1);
    c.send(2);
    c.send(3);
    c.send(4);
    r = p.recv();
    sum += r;
    debug!(r);
    r = p.recv();
    sum += r;
    debug!(r);
    r = p.recv();
    sum += r;
    debug!(r);
    r = p.recv();
    sum += r;
    debug!(r);
    c.send(5);
    c.send(6);
    c.send(7);
    c.send(8);
    r = p.recv();
    sum += r;
    debug!(r);
    r = p.recv();
    sum += r;
    debug!(r);
    r = p.recv();
    sum += r;
    debug!(r);
    r = p.recv();
    sum += r;
    debug!(r);
    assert_eq!(sum, 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8);
}
