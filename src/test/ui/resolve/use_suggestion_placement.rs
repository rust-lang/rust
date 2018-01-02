// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-cloudabi no std::path support

macro_rules! y {
    () => {}
}

mod m {
    pub const A: i32 = 0;
}

mod foo {
    #[derive(Debug)]
    pub struct Foo;

    // test whether the use suggestion isn't
    // placed into the expansion of `#[derive(Debug)]
    type Bar = Path; //~ ERROR cannot find
}

fn main() {
    y!();
    let _ = A; //~ ERROR cannot find
    foo();
}

fn foo() {
    type Dict<K, V> = HashMap<K, V>; //~ ERROR cannot find
}
