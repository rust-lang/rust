#![feature(trait_alias)]

pub struct Foo;

pub trait Bar {
    const BAZ: u8;
}

impl Bar for Foo {
    #[doc(alias = "CONST_BAZ")] //~ ERROR
    const BAZ: u8 = 0;
}

impl Foo {
    #[doc(alias = "CONST_FOO")] // ok!
    pub const FOO: u8 = 0;

    pub fn bar() -> u8 {
        Self::FOO
    }
}
