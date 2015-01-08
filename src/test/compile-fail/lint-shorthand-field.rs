// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(bad_style, unused_variables)]
#![deny(non_shorthand_field_patterns)]

struct Foo {
    x: isize,
    y: isize,
}

fn main() {
    {
        let Foo {
            x: x, //~ ERROR the `x:` in this pattern is redundant
            y: ref y, //~ ERROR the `y:` in this pattern is redundant
        } = Foo { x: 0, y: 0 };

        let Foo {
            x,
            ref y,
        } = Foo { x: 0, y: 0 };
    }

    {
        const x: isize = 1;

        match (Foo { x: 1, y: 1 }) {
            Foo { x: x, ..} => {},
            _ => {},
        }
    }

    {
        struct Bar {
            x: x,
        }

        struct x;

        match (Bar { x: x }) {
            Bar { x: x } => {},
        }
    }

    {
        struct Bar {
            x: Foo,
        }

        enum Foo { x }

        match (Bar { x: Foo::x }) {
            Bar { x: Foo::x } => {},
        }
    }
}
