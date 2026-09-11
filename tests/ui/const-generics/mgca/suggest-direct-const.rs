//! Regression test for <https://github.com/rust-lang/rust/issues/157368>
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

    pub struct Foo1([u8; core::direct_const_arg!(Bar::PUBLIC)]);
    //~^ ERROR: use of `const` in the type system not marked as direct
    pub struct Foo2([u8; core::direct_const_arg!(Bar::RESTRICTED)]);
    //~^ ERROR: use of `const` in the type system not marked as direct
    pub struct Foo3([u8; core::direct_const_arg!(Bar::PRIVATE)]);
    //~^ ERROR: use of `const` in the type system not marked as direct
}

mod top_level_item {
    pub const PUBLIC: usize = 1;
    pub(crate) const RESTRICTED: usize = 1;
    const PRIVATE: usize = 1;

    pub struct Foo1([u8; core::direct_const_arg!(PUBLIC)]);
    //~^ ERROR: use of `const` in the type system not marked as direct
    pub struct Foo2([u8; core::direct_const_arg!(RESTRICTED)]);
    //~^ ERROR: use of `const` in the type system not marked as direct
    pub struct Foo3([u8; core::direct_const_arg!(PRIVATE)]);
    //~^ ERROR: use of `const` in the type system not marked as direct
}

mod trait_item {
    pub trait Foo {
        pub const PUBLIC: usize;
        //~^ ERROR: [E0449]
        pub(crate) const RESTRICTED: usize;
        //~^ ERROR: [E0449]
        const PRIVATE: usize;
    }

    pub struct Bar<T: Foo>([u8; core::direct_const_arg!(T::PUBLIC)]);
    //~^ ERROR: use of `const` in the type system not marked as direct
    pub struct Bar2<T: Foo>([u8; core::direct_const_arg!(T::RESTRICTED)]);
    //~^ ERROR: use of `const` in the type system not marked as direct
    pub struct Bar3<T: Foo>([u8; core::direct_const_arg!(T::PRIVATE)]);
    //~^ ERROR: use of `const` in the type system not marked as direct
}

fn main() {}
