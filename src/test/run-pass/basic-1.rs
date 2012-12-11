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
use comm::Chan;
use comm::Port;
use comm::send;
use comm::recv;

fn a(c: Chan<int>) { send(c, 10); }

fn main() {
    let p = Port();
    let ch = Chan(&p);
    task::spawn(|| a(ch) );
    task::spawn(|| a(ch) );
    let mut n: int = 0;
    n = recv(p);
    n = recv(p);
    //    debug!("Finished.");
}

fn b(c: Chan<int>) {
    //    debug!("task b0");
    //    debug!("task b1");
    //    debug!("task b2");
    //    debug!("task b3");
    //    debug!("task b4");
    //    debug!("task b5");
    send(c, 10);
}
