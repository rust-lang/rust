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
struct Foo(uint, int, u8);

pub fn main() {
    use core::cmp::{Equal, Greater, Less};

    let f1 = 5;
    let f2 = -3;
    let f3 = 10;

    let a = Foo(f1, f2, f3);
    fail_unless!(a.cmp(&a) == Equal);

    {
        let bar_more = Foo(6, f2, f3);
        fail_unless!(a.cmp(&bar_more) == Less);
        fail_unless!(bar_more.cmp(&a) == Greater);
    }

    {
        let bar_less = Foo(4, f2, f3);
        fail_unless!(a.cmp(&bar_less) == Greater);
        fail_unless!(bar_less.cmp(&a) == Less);
    }

    {
        let qux_more = Foo(f1, 5, f3);
        fail_unless!(a.cmp(&qux_more) == Less);
        fail_unless!(qux_more.cmp(&a) == Greater);
    }

    {
        let qux_less = Foo(f1, -10, f3);
        fail_unless!(a.cmp(&qux_less) == Greater);
        fail_unless!(qux_less.cmp(&a) == Less);
    }

    {
        let baz_more = Foo(f1, f2, 12);
        fail_unless!(a.cmp(&baz_more) == Less);
        fail_unless!(baz_more.cmp(&a) == Greater);
    }

    {
        let baz_less = Foo(f1, f2, 3);
        fail_unless!(a.cmp(&baz_less) == Greater);
        fail_unless!(baz_less.cmp(&a) == Less);
    }
}
