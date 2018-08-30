// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that the cache results from the default method do not pollute
// the cache for the later call in `load()`.
//
// See issue #18209.

// pretty-expanded FIXME #23616

pub trait Foo {
    fn load_from() -> Box<Self>;
    fn load() -> Box<Self> {
        Foo::load_from()
    }
}

pub fn load<M: Foo>() -> Box<M> {
    Foo::load()
}

fn main() { }
