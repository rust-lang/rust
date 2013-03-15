// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[deriving(TotalOrd)]
struct Foo {
    bar: uint,
    qux: int,
    baz: u8
}

pub fn main() {
    use core::cmp::{Equal, Greater, Less};

    let a = Foo {bar: 5, qux: -3, baz: 10};
    fail_unless!(a.cmp(&a) == Equal);

    {
        let bar_more = Foo {bar: 6, ..a};
        fail_unless!(a.cmp(&bar_more) == Less);
        fail_unless!(bar_more.cmp(&a) == Greater);
    }

    {
        let qux_more = Foo {qux: 5, ..a};
        fail_unless!(a.cmp(&qux_more) == Less);
        fail_unless!(qux_more.cmp(&a) == Greater);
    }

    {
        let baz_more = Foo {baz: 12, ..a};
        fail_unless!(a.cmp(&baz_more) == Less);
        fail_unless!(baz_more.cmp(&a) == Greater);
    }
}
