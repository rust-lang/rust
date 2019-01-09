// compile-flags: -Cmetadata=aux

pub trait Foo {
    #[doc(hidden)]
    fn foo(&self) {}
}

impl Foo for i32 {}
