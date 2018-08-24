// compile-flags: -Cmetadata=aux

pub trait Foo {
    fn bar(&self);
    fn foo(&mut self) {}
}
