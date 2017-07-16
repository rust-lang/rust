// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


struct Foo;

trait HasNum {
    const NUM: isize;
}
impl HasNum for Foo {
    const NUM: isize = 1;
}

fn main() {
    assert!(match 2 {
        Foo::NUM ... 3 => true,
        _ => false,
    });
    assert!(match 0 {
        -1 ... <Foo as HasNum>::NUM => true,
        _ => false,
    });
    assert!(match 1 {
        <Foo as HasNum>::NUM ... <Foo>::NUM => true,
        _ => false,
    });
}
