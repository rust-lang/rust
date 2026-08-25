#![crate_name = "foo"]

//@ has foo/index.html '//a[@href="foo/constant.FIRSTCONST.html"]' 'foo::FIRSTCONST'
//@ has foo/index.html '//a[@href="struct.Bar.html#associatedconstant.CONST"]' 'Bar::CONST'

//! We have here [`foo::FIRSTCONST`] and [`Bar::CONST`].

pub mod foo {
    pub const FIRSTCONST: u32 = 42;
}

pub struct Bar;

impl Bar {
    pub const CONST: u32 = 42;
}
