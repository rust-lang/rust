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
use pipes::send;

fn start(c: pipes::Chan<int>, start: int, number_of_messages: int) {
    let mut i: int = 0;
    while i < number_of_messages { c.send(start + i); i += 1; }
}

fn main() {
    debug!("Check that we don't deadlock.");
    let (ch, p) = pipes::stream();
    task::try(|move ch| start(ch, 0, 10) );
    debug!("Joined task");
}
