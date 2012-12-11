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
use comm::*;

fn main() {
    let p = Port();
    let ch = Chan(&p);
    let mut y: int;

    task::spawn(|| child(ch) );
    y = recv(p);
    debug!("received 1");
    log(debug, y);
    assert (y == 10);

    task::spawn(|| child(ch) );
    y = recv(p);
    debug!("received 2");
    log(debug, y);
    assert (y == 10);
}

fn child(c: Chan<int>) { send(c, 10); }
