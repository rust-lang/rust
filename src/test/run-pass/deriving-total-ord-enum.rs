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
enum Foo {
    Bar,
    Baz {xyz: uint},
    Qux(int, u8),
}

pub fn main() {
    use core::cmp::{Equal, Greater, Less};

    let bar = Bar;
    fail_unless!(bar.cmp(&bar) == Equal);

    let baz_4 = Baz {xyz: 4};
    fail_unless!(baz_4.cmp(&baz_4) == Equal);

    fail_unless!(bar.cmp(&baz_4) == Less);
    fail_unless!(baz_4.cmp(&bar) == Greater);

    {
        let baz_5 = Baz {xyz: 5};
        
        fail_unless!(baz_4.cmp(&baz_5) == Less);
        fail_unless!(baz_5.cmp(&baz_4) == Greater);
    }

    let qux_0_6 = Qux(0, 6);
    fail_unless!(qux_0_6.cmp(&qux_0_6) == Equal);

    fail_unless!(bar.cmp(&qux_0_6) == Less);
    fail_unless!(qux_0_6.cmp(&bar) == Greater);

    fail_unless!(baz_4.cmp(&qux_0_6) == Less);
    fail_unless!(qux_0_6.cmp(&baz_4) == Greater);
}
