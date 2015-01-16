// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:no_method_suggested_traits.rs

extern crate no_method_suggested_traits;

mod foo {
    trait Bar {
        fn method(&self) {}

        fn method2(&self) {}
    }

    impl Bar for u32 {}

    impl Bar for char {}
}

fn main() {
    1u32.method();
    //~^ ERROR does not implement
    //~^^ HELP the following traits are implemented and define a method `method`
    //~^^^ HELP `foo::Bar`
    //~^^^^ HELP `no_method_suggested_traits::foo::PubPub`

    'a'.method();
    //~^ ERROR does not implement
    //~^^ HELP the following trait is implemented and defines a method `method`
    //~^^^ HELP `foo::Bar`

    1i32.method();
    //~^ ERROR does not implement
    //~^^ HELP the following trait is implemented and defines a method `method`
    //~^^^ HELP `no_method_suggested_traits::foo::PubPub`

    1u64.method();
    //~^ ERROR does not implement
    //~^^ HELP the following traits define a method `method`
    //~^^^ HELP `foo::Bar`
    //~^^^^ HELP `no_method_suggested_traits::foo::PubPub`
    //~^^^^^ HELP `no_method_suggested_traits::reexport::Reexported`
    //~^^^^^^ HELP `no_method_suggested_traits::bar::PubPriv`
    //~^^^^^^^ HELP `no_method_suggested_traits::qux::PrivPub`
    //~^^^^^^^^ HELP `no_method_suggested_traits::quz::PrivPriv`

    1u64.method2();
    //~^ ERROR does not implement
    //~^^ HELP the following trait defines a method `method2`
    //~^^^ HELP `foo::Bar`
    1u64.method3();
    //~^ ERROR does not implement
    //~^^ HELP the following trait defines a method `method3`
    //~^^^ HELP `no_method_suggested_traits::foo::PubPub`
}
