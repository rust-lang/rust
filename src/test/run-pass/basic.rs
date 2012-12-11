// -*- rust -*-
// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


extern mod std;
use comm::send;
use comm::Chan;
use comm::recv;

fn a(c: Chan<int>) {
    if true {
        debug!("task a");
        debug!("task a");
        debug!("task a");
        debug!("task a");
        debug!("task a");
    }
    send(c, 10);
}

fn k(x: int) -> int { return 15; }

fn g(x: int, y: ~str) -> int {
    log(debug, x);
    log(debug, y);
    let z: int = k(1);
    return z;
}

fn main() {
    let mut n: int = 2 + 3 * 7;
    let s: ~str = ~"hello there";
    let p = comm::Port();
    let ch = comm::Chan(&p);
    task::spawn(|| a(ch) );
    task::spawn(|| b(ch) );
    let mut x: int = 10;
    x = g(n, s);
    log(debug, x);
    n = recv(p);
    n = recv(p);
    debug!("children finished, root finishing");
}

fn b(c: Chan<int>) {
    if true {
        debug!("task b");
        debug!("task b");
        debug!("task b");
        debug!("task b");
        debug!("task b");
        debug!("task b");
    }
    send(c, 10);
}
