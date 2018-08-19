// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-pass
// run-pass

enum Foo {
    X = 42,
    Y = Foo::X as isize - 3,
}

enum Bar {
    X,
    Y = Bar::X as isize + 2,
}

enum Boo {
    X = Boo::Y as isize * 2,
    Y = 9,
}

fn main() {
    assert_eq!(Foo::X as isize, 42);
    assert_eq!(Foo::Y as isize, 39);
    assert_eq!(Bar::X as isize, 0);
    assert_eq!(Bar::Y as isize, 2);
    assert_eq!(Boo::X as isize, 18);
    assert_eq!(Boo::Y as isize, 9);
}
