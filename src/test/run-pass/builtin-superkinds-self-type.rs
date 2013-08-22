// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Tests the ability for the Self type in default methods to use
// capabilities granted by builtin kinds as supertraits.

use std::comm;

trait Foo : Send {
    fn foo(self, chan: comm::Chan<Self>) {
        chan.send(self);
    }
}

impl <T: Send> Foo for T { }

fn main() {
    let (p,c) = comm::stream();
    1193182.foo(c);
    assert!(p.recv() == 1193182);
}
