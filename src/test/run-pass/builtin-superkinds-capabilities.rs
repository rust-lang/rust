// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait Foo : Send { }

impl <T: Send> Foo for T { }

fn foo<T: Foo>(val: T, chan: std::comm::Chan<T>) {
    chan.send(val);
}

fn main() {
    let (p,c) = std::comm::stream();
    foo(31337, c);
    assert!(p.recv() == 31337);
}
