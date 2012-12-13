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


fn a(c: core::oldcomm::Chan<int>) {
    debug!("task a0");
    debug!("task a1");
    core::oldcomm::send(c, 10);
}

fn main() {
    let p = core::oldcomm::Port();
    let ch = core::oldcomm::Chan(&p);
    task::spawn(|| a(ch) );
    task::spawn(|| b(ch) );
    let mut n: int = 0;
    n = core::oldcomm::recv(p);
    n = core::oldcomm::recv(p);
    debug!("Finished.");
}

fn b(c: core::oldcomm::Chan<int>) {
    debug!("task b0");
    debug!("task b1");
    debug!("task b2");
    debug!("task b2");
    debug!("task b3");
    core::oldcomm::send(c, 10);
}
