// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// @has structfields/Foo.t.html
// @has - struct.Foo.html
// @has structfields/struct.Foo.html
pub struct Foo {
    // @has - //pre "pub a: ()"
    pub a: (),
    // @has - //pre "// some fields omitted"
    // @!has - //pre "b: ()"
    b: (),
    // @!has - //pre "c: usize"
    #[doc(hidden)]
    c: usize,
    // @has - //pre "pub d: usize"
    pub d: usize,
}

// @has structfields/Bar.t.html
// @has - struct.Bar.html
// @has structfields/struct.Bar.html
pub struct Bar {
    // @has - //pre "pub a: ()"
    pub a: (),
    // @!has - //pre "// some fields omitted"
}

// @has structfields/Qux.t.html
// @has - enum.Qux.html
// @has structfields/enum.Qux.html
pub enum Qux {
    Quz {
        // @has - //pre "a: ()"
        a: (),
        // @!has - //pre "b: ()"
        #[doc(hidden)]
        b: (),
        // @has - //pre "c: usize"
        c: usize,
        // @has - //pre "// some fields omitted"
    },
}

// @has structfields/struct.Baz.html //pre "pub struct Baz { /* fields omitted */ }"
pub struct Baz {
    x: u8,
    #[doc(hidden)]
    pub y: u8,
}

// @has structfields/struct.Quux.html //pre "pub struct Quux {}"
pub struct Quux {}
