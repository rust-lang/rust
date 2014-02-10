// xfail-test

// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub struct send_packet<T> {
  p: T
}


mod pingpong {
    use send_packet;
    pub type ping = send_packet<pong>;
    pub struct pong(send_packet<ping>);
    //~^ ERROR illegal recursive enum type; wrap the inner value in a box to make it representable
}

fn main() {}
