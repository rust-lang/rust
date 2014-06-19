// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

enum X {
    Foo(uint),
    Bar(bool)
}

fn main() {
    let x = match Foo(42) {
        Foo(..) => 1,
        _ if true => 0,
        Bar(..) => fail!("Oh dear")
    };
    assert_eq!(x, 1);

    let x = match Foo(42) {
        _ if true => 0,
        Foo(..) => 1,
        Bar(..) => fail!("Oh dear")
    };
    assert_eq!(x, 0);
}
