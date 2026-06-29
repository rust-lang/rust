//! Regression test for <https://github.com/rust-lang/rust/issues/157368>
//! Tests suggesting `pub type const` instead of `type pub const`.
//@ run-rustfix
#![feature(min_generic_const_args, inherent_associated_types)]
#![allow(dead_code)]

mod impl_item {
    pub struct Bar;
    impl Bar {
        pub const PUBLIC: usize = 1;
        pub(crate) const RESTRICTED: usize = 1;
        const PRIVATE: usize = 1;
    }

    pub struct Foo1([u8; Bar::PUBLIC]);
    //~^ ERROR: use of `const` in the type system not defined as `type const`
    pub struct Foo2([u8; Bar::RESTRICTED]);
    //~^ ERROR: use of `const` in the type system not defined as `type const`
    pub struct Foo3([u8; Bar::PRIVATE]);
    //~^ ERROR: use of `const` in the type system not defined as `type const`
}

mod top_level_item {
    pub const PUBLIC: usize = 1;
    pub(crate) const RESTRICTED: usize = 1;
    const PRIVATE: usize = 1;

    pub struct Foo1([u8; PUBLIC]);
    //~^ ERROR: use of `const` in the type system not defined as `type const`
    pub struct Foo2([u8; RESTRICTED]);
    //~^ ERROR: use of `const` in the type system not defined as `type const`
    pub struct Foo3([u8; PRIVATE]);
    //~^ ERROR: use of `const` in the type system not defined as `type const`
}

mod trait_item {
    pub trait Foo {
        pub const PUBLIC: usize;
        //~^ ERROR: [E0449]
        pub(crate) const RESTRICTED: usize;
        //~^ ERROR: [E0449]
        const PRIVATE: usize;
    }

    pub struct Bar<T: Foo>([u8; T::PUBLIC]);
    //~^ ERROR: use of `const` in the type system not defined as `type const`
    pub struct Bar2<T: Foo>([u8; T::RESTRICTED]);
    //~^ ERROR: use of `const` in the type system not defined as `type const`
    pub struct Bar3<T: Foo>([u8; T::PRIVATE]);
    //~^ ERROR: use of `const` in the type system not defined as `type const`
}

fn main() {}
