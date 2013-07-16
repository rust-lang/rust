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

use std::task;

fn x(s: ~str, n: int) {
    info!(s);
    info!(n);
}

pub fn main() {
    task::spawn(|| x(~"hello from first spawned fn", 65) );
    task::spawn(|| x(~"hello from second spawned fn", 66) );
    task::spawn(|| x(~"hello from third spawned fn", 67) );
    let mut i: int = 30;
    while i > 0 { i = i - 1; info!("parent sleeping"); task::yield(); }
}
