//@ compile-flags: -Cmetadata=aux

pub trait Foo {
    #[doc(hidden)]
    fn foo(&self) {}
    fn not_hidden(&self) {}
}

impl Foo for i32 {}
