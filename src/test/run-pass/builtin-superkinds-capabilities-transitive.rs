// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Tests "transitivity" of super-builtin-kinds on traits. Here, if
// we have a Foo, we know we have a Bar, and if we have a Bar, we
// know we have a Send. So if we have a Foo we should know we have
// a Send. Basically this just makes sure rustc is using
// each_bound_trait_and_supertraits in type_contents correctly.

use std::comm;

trait Bar : Send { }
trait Foo : Bar { }

impl <T: Send> Foo for T { }
impl <T: Send> Bar for T { }

fn foo<T: Foo>(val: T, chan: comm::Chan<T>) {
    chan.send(val);
}

fn main() {
    let (p,c) = comm::stream();
    foo(31337, c);
    assert!(p.recv() == 31337);
}
