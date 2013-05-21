// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use extra;

struct Deserializer : extra::serialization::deserializer{ //~ ERROR obsolete syntax: class traits
    x: ()
}

struct Foo {
    a: ()
}

fn deserialize_foo<__D: extra::serialization::deserializer>(__d: __D) {
}

fn main() { let des = Deserializer(); let foo = deserialize_foo(des); }
