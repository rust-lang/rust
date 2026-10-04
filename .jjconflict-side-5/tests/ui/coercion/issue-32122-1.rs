//@ run-rustfix
use std::ops::Deref;

struct Foo(u8);

impl Deref for Foo {
    type Target = u8;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

fn main() {
    let a = Foo(0);
    // Should suggest `&*` when coercing &ty to *const ty
    let _: *const u8 = &a; //~ ERROR mismatched types
}
