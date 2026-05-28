#![crate_type = "lib"]

#[repr(u32)]
pub enum Foo {
    Foo = Private::Variant as u32
}

#[repr(u8)]
enum Private {
    Variant = 42
}

#[inline(always)]
pub fn foo() -> Foo {
    Foo::Foo
}
