//@ run-rustfix
use std::ops::Deref;
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
fn main() {
    let a = Emm(Foo(Bar(0)));
    // Should suggest `&***` even when deref is pretty deep
    let _: *const u8 = &a; //~ ERROR mismatched types
}
