// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-pretty pretty-printing is unhygienic

#![feature(decl_macro)]

mod foo {
    struct S { x: u32 }
    struct T(u32);

    pub macro m($S:ident, $x:ident) {{
        struct $S {
            $x: u32,
            x: i32,
        }

        let s = S { x: 0 }; //~ ERROR type `foo::S` is private
        let _ = s.x; //~ ERROR type `foo::S` is private

        let t = T(0); //~ ERROR type `foo::T` is private
        let _ = t.0; //~ ERROR type `foo::T` is private

        let s = $S { $x: 0, x: 1 };
        assert_eq!((s.$x, s.x), (0, 1));
        s
    }}
}

fn main() {
    let s = foo::m!(S, x);
    assert_eq!(s.x, 0);
}
