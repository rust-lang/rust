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

struct Foo;
enum Bar { X }

mod foo {
    trait Bar {
        fn method(&self) {}

        fn method2(&self) {}
    }

    impl Bar for u32 {}

    impl Bar for char {}
}

fn main() {
    // test the values themselves, and autoderef.


    1u32.method();
    //~^ HELP following traits are implemented but not in scope, perhaps add a `use` for one of them
    //~^^ ERROR no method named
    //~^^^ HELP `use foo::Bar;`
    //~^^^^ HELP `use no_method_suggested_traits::foo::PubPub;`
    std::rc::Rc::new(&mut Box::new(&1u32)).method();
    //~^ HELP following traits are implemented but not in scope, perhaps add a `use` for one of them
    //~^^ ERROR no method named
    //~^^^ HELP `use foo::Bar;`
    //~^^^^ HELP `use no_method_suggested_traits::foo::PubPub;`

    'a'.method();
    //~^ ERROR no method named
    //~^^ HELP the following trait is implemented but not in scope, perhaps add a `use` for it:
    //~^^^ HELP `use foo::Bar;`
    std::rc::Rc::new(&mut Box::new(&'a')).method();
    //~^ ERROR no method named
    //~^^ HELP the following trait is implemented but not in scope, perhaps add a `use` for it:
    //~^^^ HELP `use foo::Bar;`

    1i32.method();
    //~^ ERROR no method named
    //~^^ HELP the following trait is implemented but not in scope, perhaps add a `use` for it:
    //~^^^ HELP `use no_method_suggested_traits::foo::PubPub;`
    std::rc::Rc::new(&mut Box::new(&1i32)).method();
    //~^ ERROR no method named
    //~^^ HELP the following trait is implemented but not in scope, perhaps add a `use` for it:
    //~^^^ HELP `use no_method_suggested_traits::foo::PubPub;`

    Foo.method();
    //~^ ERROR no method named
    //~^^ HELP following traits define an item `method`, perhaps you need to implement one of them
    //~^^^ HELP `foo::Bar`
    //~^^^^ HELP `no_method_suggested_traits::foo::PubPub`
    //~^^^^^ HELP `no_method_suggested_traits::Reexported`
    //~^^^^^^ HELP `no_method_suggested_traits::bar::PubPriv`
    //~^^^^^^^ HELP `no_method_suggested_traits::qux::PrivPub`
    //~^^^^^^^^ HELP `no_method_suggested_traits::quz::PrivPriv`
    std::rc::Rc::new(&mut Box::new(&Foo)).method();
    //~^ ERROR no method named
    //~^^ HELP following traits define an item `method`, perhaps you need to implement one of them
    //~^^^ HELP `foo::Bar`
    //~^^^^ HELP `no_method_suggested_traits::foo::PubPub`
    //~^^^^^ HELP `no_method_suggested_traits::Reexported`
    //~^^^^^^ HELP `no_method_suggested_traits::bar::PubPriv`
    //~^^^^^^^ HELP `no_method_suggested_traits::qux::PrivPub`
    //~^^^^^^^^ HELP `no_method_suggested_traits::quz::PrivPriv`

    1u64.method2();
    //~^ ERROR no method named
    //~^^ HELP the following trait defines an item `method2`, perhaps you need to implement it
    //~^^^ HELP `foo::Bar`
    std::rc::Rc::new(&mut Box::new(&1u64)).method2();
    //~^ ERROR no method named
    //~^^ HELP the following trait defines an item `method2`, perhaps you need to implement it
    //~^^^ HELP `foo::Bar`

    no_method_suggested_traits::Foo.method2();
    //~^ ERROR no method named
    //~^^ HELP following trait defines an item `method2`, perhaps you need to implement it
    //~^^^ HELP `foo::Bar`
    std::rc::Rc::new(&mut Box::new(&no_method_suggested_traits::Foo)).method2();
    //~^ ERROR no method named
    //~^^ HELP following trait defines an item `method2`, perhaps you need to implement it
    //~^^^ HELP `foo::Bar`
    no_method_suggested_traits::Bar::X.method2();
    //~^ ERROR no method named
    //~^^ HELP following trait defines an item `method2`, perhaps you need to implement it
    //~^^^ HELP `foo::Bar`
    std::rc::Rc::new(&mut Box::new(&no_method_suggested_traits::Bar::X)).method2();
    //~^ ERROR no method named
    //~^^ HELP following trait defines an item `method2`, perhaps you need to implement it
    //~^^^ HELP `foo::Bar`

    Foo.method3();
    //~^ ERROR no method named
    //~^^ HELP following trait defines an item `method3`, perhaps you need to implement it
    //~^^^ HELP `no_method_suggested_traits::foo::PubPub`
    std::rc::Rc::new(&mut Box::new(&Foo)).method3();
    //~^ ERROR no method named
    //~^^ HELP following trait defines an item `method3`, perhaps you need to implement it
    //~^^^ HELP `no_method_suggested_traits::foo::PubPub`
    Bar::X.method3();
    //~^ ERROR no method named
    //~^^ HELP following trait defines an item `method3`, perhaps you need to implement it
    //~^^^ HELP `no_method_suggested_traits::foo::PubPub`
    std::rc::Rc::new(&mut Box::new(&Bar::X)).method3();
    //~^ ERROR no method named
    //~^^ HELP following trait defines an item `method3`, perhaps you need to implement it
    //~^^^ HELP `no_method_suggested_traits::foo::PubPub`

    // should have no help:
    1_usize.method3(); //~ ERROR no method named
    std::rc::Rc::new(&mut Box::new(&1_usize)).method3(); //~ ERROR no method named
    no_method_suggested_traits::Foo.method3();  //~ ERROR no method named
    std::rc::Rc::new(&mut Box::new(&no_method_suggested_traits::Foo)).method3();
    //~^ ERROR no method named
    no_method_suggested_traits::Bar::X.method3();  //~ ERROR no method named
    std::rc::Rc::new(&mut Box::new(&no_method_suggested_traits::Bar::X)).method3();
    //~^ ERROR no method named
}
