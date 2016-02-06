// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:rustdoc-default-impl.rs
// ignore-cross-compile

extern crate rustdoc_default_impl as foo;

pub use foo::bar;

pub fn wut<T: bar::Bar>() {
}

/* !search-index
{
    "default_impl": {
        "default_impl::Foo::test": [
            "Method()"
        ],
        "default_impl::bar": [
            "Module"
        ],
        "default_impl::bar::Bar": [
            "Trait"
        ],
        "default_impl::bar::Foo": [
            "Trait"
        ],
        "default_impl::bar::Foo::foo": [
            "Method()"
        ],
        "default_impl::bar::TypeId": [
            "Struct"
        ],
        "default_impl::wut": [
            "Function()"
        ],
        "rustdoc_default_impl::bar::TypeId::of": [
            "Method(typeid) -> typeid"
        ]
    }
}
*/
