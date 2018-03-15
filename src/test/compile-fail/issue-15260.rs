// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct Foo {
    a: usize,
}

fn main() {
    let Foo {
        a: _, //~ NOTE first use of `a`
        a: _
        //~^ ERROR field `a` bound multiple times in the pattern
        //~| NOTE multiple uses of `a` in pattern
    } = Foo { a: 29 };

    let Foo {
        a, //~ NOTE first use of `a`
        a: _
        //~^ ERROR field `a` bound multiple times in the pattern
        //~| NOTE multiple uses of `a` in pattern
    } = Foo { a: 29 };

    let Foo {
        a,
        //~^ NOTE first use of `a`
        //~| NOTE first use of `a`
        a: _,
        //~^ ERROR field `a` bound multiple times in the pattern
        //~| NOTE multiple uses of `a` in pattern
        a: x
        //~^ ERROR field `a` bound multiple times in the pattern
        //~| NOTE multiple uses of `a` in pattern
    } = Foo { a: 29 };
}
