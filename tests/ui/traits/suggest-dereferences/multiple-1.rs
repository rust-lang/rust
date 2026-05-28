use std::ops::{Deref, DerefMut};

trait Happy {}
struct LDM;
impl Happy for &mut LDM {}

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
impl DerefMut for Foo {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
impl DerefMut for Bar {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
impl DerefMut for Baz {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}


fn foo<T>(_: T) where T: Happy {}

fn main() {
    // Currently the compiler doesn't try to suggest dereferences for situations
    // where DerefMut involves. So this test is meant to ensure compiler doesn't
    // generate incorrect help message.
    let mut baz = Baz(Bar(Foo(LDM)));
    foo(&mut baz);
    //~^ ERROR the trait bound `&mut Baz: Happy` is not satisfied
}
