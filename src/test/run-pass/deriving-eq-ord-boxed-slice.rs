// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[derive(PartialEq, PartialOrd, Eq, Ord)]
struct Foo(Box<[u8]>);

pub fn main() {
    // FIXME (#22405): Replace `Box::new` with `box` here when/if possible.
    let a = Foo(Box::new([0, 1, 2]));
    let b = Foo(Box::new([0, 1, 2]));
    assert!(a == b);
    println!("{}", a != b);
    println!("{}", a < b);
    println!("{}", a <= b);
    println!("{}", a == b);
    println!("{}", a > b);
    println!("{}", a >= b);
}
