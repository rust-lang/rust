//@ run-rustfix
use std::ops::Deref;
use std::ops::DerefMut;
struct Bar(u8);
struct Foo(Bar);
struct Emm(Foo);
impl Deref for Bar{
    type Target = u8;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl Deref for Foo {
    type Target = Bar;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl Deref for Emm {
    type Target = Foo;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for Bar{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
impl DerefMut for Foo {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
impl DerefMut for Emm {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
fn main() {
    // Suggest dereference with arbitrary mutability
    let a = Emm(Foo(Bar(0)));
    let _: *const u8 = &a; //~ ERROR mismatched types

    let mut a = Emm(Foo(Bar(0)));
    let _: *mut u8 = &a; //~ ERROR mismatched types

    let a = Emm(Foo(Bar(0)));
    let _: *const u8 = &mut a; //~ ERROR mismatched types

    let mut a = Emm(Foo(Bar(0)));
    let _: *mut u8 = &mut a; //~ ERROR mismatched types
}
