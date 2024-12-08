#![feature(lang_items, start, no_core)]
#![no_core] // makes debugging this test *a lot* easier (during resolve)

#[lang="sized"]
pub trait Sized {}

#[lang="copy"]
pub trait Copy {}

#[lang="deref"]
pub trait Deref {
    type Target;
}

#[lang="legacy_receiver"]
pub trait LegacyReceiver: Deref {}

impl<'a, T> Deref for &'a T {
    type Target = T;
}

impl<'a, T> LegacyReceiver for &'a T {}

mod bar {
    // shouldn't bring in too much
    pub use self::glob::*;

    // can't publicly re-export private items
    pub use self::baz::{foo, bar};

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

    impl B for isize { fn foo() -> isize { 3 } }

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

        pub fn foo() {}
        pub fn bar() {}
    }

    extern "C" {
        fn epriv();
        pub fn epub();
    }

    fn test() {
        self::Enum::Pub;
        unsafe {
            epriv();
            epub();
        }
        self::baz::A;
        self::baz::A::foo();
        self::baz::A::bar(); //~ ERROR: associated function `bar` is private
        self::baz::A.foo2();

        // this used to cause an ICE in privacy traversal.
        super::gpub();
    }

    mod glob {
        pub fn gpub() {}
        fn gpriv() {}
    }
}

pub fn gpub() {}

fn lol() {
    bar::A;
    bar::A::foo();
    bar::A::bar(); //~ ERROR: associated function `bar` is private
    bar::A.foo2();
}

mod foo {
    fn test() {
        ::bar::A::foo();
        ::bar::A::bar();        //~ ERROR: associated function `bar` is private
        ::bar::A.foo2();
        ::bar::baz::A::foo();   //~ ERROR: module `baz` is private
        ::bar::baz::A::bar();   //~ ERROR: module `baz` is private
                                //~^ ERROR: associated function `bar` is private
        ::bar::baz::A.foo2();   //~ ERROR: module `baz` is private
        ::bar::baz::A.bar2();   //~ ERROR: module `baz` is private
                                //~^ ERROR: method `bar2` is private

        let _: isize =
        ::bar::B::foo();        //~ ERROR: trait `B` is private
        ::lol();

        ::bar::Enum::Pub;

        unsafe {
            ::bar::epriv(); //~ ERROR: function `epriv` is private
            ::bar::epub();
        }

        ::bar::foo();
        ::bar::bar();

        ::bar::gpub();

        ::bar::baz::foo(); //~ ERROR: module `baz` is private
        ::bar::baz::bar(); //~ ERROR: module `baz` is private
    }

    fn test2() {
        use bar::baz::{foo, bar};
        //~^ ERROR: module `baz` is private
        //~| ERROR: module `baz` is private

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
    use self::foo::i::A; //~ ERROR: module `i` is private

    pub mod foo {
        pub use self::i::A as foo;

        mod i {
            pub struct A;
        }
    }
}

#[start] fn main(_: isize, _: *const *const u8) -> isize { 3 }
