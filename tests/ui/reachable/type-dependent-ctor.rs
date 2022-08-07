// Verify that we do not warn on type-dependent constructors (`Self::A` below).
//@ check-pass
#![deny(unreachable_code)]

enum Void {}

enum Foo {
    A(Void),
}

impl Foo {
    fn wrap(x: Void) -> Self {
        Self::A(x)
    }

    fn make() -> Self {
        Self::A(produce())
    }
}

fn produce() -> Void {
    panic!()
}

fn main() {}
