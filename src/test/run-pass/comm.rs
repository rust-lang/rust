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
use comm::send;
use comm::recv;

fn main() {
    let p = comm::Port();
    let ch = comm::Chan(&p);
    let t = task::spawn(|| child(ch) );
    let y = recv(p);
    error!("received");
    log(error, y);
    assert (y == 10);
}

fn child(c: Chan<int>) {
    error!("sending");
    send(c, 10);
    error!("value sent");
}
