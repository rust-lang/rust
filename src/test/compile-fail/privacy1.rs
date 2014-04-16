// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(globs)]
#![no_std] // makes debugging this test *a lot* easier (during resolve)

mod bar {
    // shouln't bring in too much
    pub use self::glob::*;

    // can't publicly re-export private items
    pub use self::baz::{foo, bar};
    //~^ ERROR: function `bar` is private

    pub use self::private::ppriv;
    //~^ ERROR: function `ppriv` is private

    pub struct A;
    impl A {
        pub fn foo() {}
        fn bar() {}

        pub fn foo2(&self) {}
        fn bar2(&self) {}
    }

    trait B {
        fn foo() -> Self;
    }

    impl B for int { fn foo() -> int { 3 } }

    pub enum Enum {
        Pub
    }

    mod baz {
        pub struct A;
        impl A {
            pub fn foo() {}
            fn bar() {}

            pub fn foo2(&self) {}
            fn bar2(&self) {}
        }

        // both of these are re-exported by `bar`, but only one should be
        // validly re-exported
        pub fn foo() {}
        fn bar() {}
    }

    extern {
        fn epriv();
        pub fn epub();
    }

    fn test() {
        self::Pub;
        unsafe {
            epriv();
            epub();
        }
        self::baz::A;
        self::baz::A::foo();
        self::baz::A::bar(); //~ ERROR: method `bar` is private
        self::baz::A.foo2();
        self::baz::A.bar2(); //~ ERROR: method `bar2` is private

        // this used to cause an ICE in privacy traversal.
        super::gpub();
    }

    mod glob {
        pub fn gpub() {}
        fn gpriv() {}
    }

    mod private {
        fn ppriv() {}
    }
}

pub fn gpub() {}

fn lol() {
    bar::A;
    bar::A::foo();
    bar::A::bar(); //~ ERROR: method `bar` is private
    bar::A.foo2();
    bar::A.bar2(); //~ ERROR: method `bar2` is private
}

mod foo {
    fn test() {
        ::bar::A::foo();
        ::bar::A::bar();        //~ ERROR: method `bar` is private
        ::bar::A.foo2();
        ::bar::A.bar2();        //~ ERROR: method `bar2` is private
        ::bar::baz::A::foo();   //~ ERROR: method `foo` is inaccessible
                                //~^ NOTE: module `baz` is private
        ::bar::baz::A::bar();   //~ ERROR: method `bar` is private
        ::bar::baz::A.foo2();   //~ ERROR: struct `A` is inaccessible
                                //~^ NOTE: module `baz` is private
        ::bar::baz::A.bar2();   //~ ERROR: struct `A` is inaccessible
                                //~^ ERROR: method `bar2` is private
                                //~^^ NOTE: module `baz` is private

        let _: int =
        ::bar::B::foo();        //~ ERROR: method `foo` is inaccessible
                                //~^ NOTE: trait `B` is private
        ::lol();

        ::bar::Pub;

        unsafe {
            ::bar::epriv(); //~ ERROR: function `epriv` is private
            ::bar::epub();
        }

        ::bar::foo();
        ::bar::bar();

        ::bar::gpub();

        ::bar::baz::foo(); //~ ERROR: function `foo` is inaccessible
                           //~^ NOTE: module `baz` is private
        ::bar::baz::bar(); //~ ERROR: function `bar` is private
    }

    fn test2() {
        use bar::baz::{foo, bar};
        //~^ ERROR: function `foo` is inaccessible
        //~^^ ERROR: function `bar` is private
        foo();
        bar();
    }

    fn test3() {
        use bar::baz;
        //~^ ERROR: module `baz` is private
    }

    fn test4() {
        use bar::{foo, bar};
        foo();
        bar();
    }

    fn test5() {
        use bar;
        bar::foo();
        bar::bar();
    }

    impl ::bar::B for f32 { fn foo() -> f32 { 1.0 } }
    //~^ ERROR: trait `B` is private
}

pub mod mytest {
    // Even though the inner `A` struct is a publicly exported item (usable from
    // external crates through `foo::foo`, it should not be accessible through
    // its definition path (which has the private `i` module).
    use self::foo::foo;
    use self::foo::i::A; //~ ERROR: type `A` is inaccessible
                         //~^ NOTE: module `i` is private

    pub mod foo {
        pub use foo = self::i::A;

        mod i {
            pub struct A;
        }
    }
}

#[start] fn main(_: int, _: **u8) -> int { 3 }
