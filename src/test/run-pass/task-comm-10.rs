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
#[legacy_modes];

extern mod std;

fn start(c: pipes::Chan<pipes::Chan<~str>>) {
    let (ch, p) = pipes::stream();
    c.send(move ch);

    let mut a;
    let mut b;
    a = p.recv();
    assert a == ~"A";
    log(error, a);
    b = p.recv();
    assert b == ~"B";
    log(error, move b);
}

fn main() {
    let (ch, p) = pipes::stream();
    let child = task::spawn(|move ch| start(ch) );

    let c = p.recv();
    c.send(~"A");
    c.send(~"B");
    task::yield();
}
