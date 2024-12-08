//@ run-rustfix
use std::ops::Deref;

trait Happy {}
struct LDM;
impl Happy for &LDM {}

struct Foo(LDM);
struct Bar(Foo);
struct Baz(Bar);
impl Deref for Foo {
    type Target = LDM;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl Deref for Bar {
    type Target = Foo;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl Deref for Baz {
    type Target = Bar;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

fn foo<T>(_: T) where T: Happy {}

fn main() {
    let baz = Baz(Bar(Foo(LDM)));
    foo(&baz);
    //~^ ERROR the trait bound `&Baz: Happy` is not satisfied
}
