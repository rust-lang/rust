// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(universal_impl_trait)]
#![crate_name = "foo"]

// @has foo/fn.foo.html
// @has - //pre 'foo('
// @matches - '_x: impl <a class="trait" href="[^"]+/trait\.Clone\.html"'
// @matches - '_z: .+impl.+trait\.Copy\.html.+, impl.+trait\.Clone\.html'
pub fn foo(_x: impl Clone, _y: i32, _z: (impl Copy, impl Clone)) {
}

pub trait Trait {
    // @has foo/trait.Trait.html
    // @has - 'method</a>('
    // @matches - '_x: impl <a class="trait" href="[^"]+/trait\.Debug\.html"'
    fn method(&self, _x: impl std::fmt::Debug) {
    }
}

pub struct S<T>(T);

impl<T> S<T> {
    // @has foo/struct.S.html
    // @has - 'bar</a>('
    // @matches - '_bar: impl <a class="trait" href="[^"]+/trait\.Copy\.html"'
    pub fn bar(_bar: impl Copy) {
    }

    // @has - 'baz</a>('
    // @matches - '_baz:.+struct\.S\.html.+impl .+trait\.Clone\.html'
    pub fn baz(_baz: S<impl Clone>) {
    }
}

// @has - 'method</a>('
// @matches - '_x: impl <a class="trait" href="[^"]+/trait\.Debug\.html"'
impl<T> Trait for S<T> {}
