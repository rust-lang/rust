// Verify that we do not warn on type-dependent constructors (`Self::A` below).
#![deny(unreachable_code)]

enum Void {}

enum Foo {
    A(Void),
}

impl Foo {
    fn wrap(x: Void) -> Self {
        Self::A(x)
        //~^ ERROR unreachable call
    }

    fn make() -> Self {
        Self::A(produce())
        //~^ ERROR unreachable call
    }
}

fn produce() -> Void {
    panic!()
}

fn main() {}
