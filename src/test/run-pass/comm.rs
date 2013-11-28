// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::task;

pub fn main() {
    let (p, ch) = stream();
    let _t = task::spawn(proc() child(&ch));
    let y = p.recv();
    error!("received");
    error!("{:?}", y);
    assert_eq!(y, 10);
}

fn child(c: &Chan<int>) {
    error!("sending");
    c.send(10);
    error!("value sent");
}
