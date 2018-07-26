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
    pub trait Bar {
        fn method(&self) {}

        fn method2(&self) {}
    }

    impl Bar for u32 {}

    impl Bar for char {}
}

fn main() {
    // test the values themselves, and autoderef.


    1u32.method();
    //~^ ERROR no method named
    //~|items from traits can only be used if the trait is in scope
    std::rc::Rc::new(&mut Box::new(&1u32)).method();
    //~^items from traits can only be used if the trait is in scope
    //~| ERROR no method named `method` found for type

    'a'.method();
    //~^ ERROR no method named
    std::rc::Rc::new(&mut Box::new(&'a')).method();
    //~^ ERROR no method named

    1i32.method();
    //~^ ERROR no method named
    std::rc::Rc::new(&mut Box::new(&1i32)).method();
    //~^ ERROR no method named

    Foo.method();
    //~^ ERROR no method named
    std::rc::Rc::new(&mut Box::new(&Foo)).method();
    //~^ ERROR no method named

    1u64.method2();
    //~^ ERROR no method named
    std::rc::Rc::new(&mut Box::new(&1u64)).method2();
    //~^ ERROR no method named

    no_method_suggested_traits::Foo.method2();
    //~^ ERROR no method named
    std::rc::Rc::new(&mut Box::new(&no_method_suggested_traits::Foo)).method2();
    //~^ ERROR no method named
    no_method_suggested_traits::Bar::X.method2();
    //~^ ERROR no method named
    std::rc::Rc::new(&mut Box::new(&no_method_suggested_traits::Bar::X)).method2();
    //~^ ERROR no method named

    Foo.method3();
    //~^ ERROR no method named
    std::rc::Rc::new(&mut Box::new(&Foo)).method3();
    //~^ ERROR no method named
    Bar::X.method3();
    //~^ ERROR no method named
    std::rc::Rc::new(&mut Box::new(&Bar::X)).method3();
    //~^ ERROR no method named

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
