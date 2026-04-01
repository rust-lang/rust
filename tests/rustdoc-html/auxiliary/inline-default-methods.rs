//@ compile-flags: -Cmetadata=aux

pub trait Foo {
    fn bar(&self);
    fn foo(&mut self) {}
}

pub trait Bar {
    fn bar(&self);
    fn foo1(&mut self) {}
    fn foo2(&mut self) {}
}

pub trait Baz {
    fn bar1(&self);
    fn bar2(&self);
    fn foo(&mut self) {}
}
