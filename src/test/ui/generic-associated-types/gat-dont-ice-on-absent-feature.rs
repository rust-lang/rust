// rust-lang/rust#60654: Do not ICE on an attempt to use GATs that is
// missing the feature gate.

struct Foo;

impl Iterator for Foo {
    type Item<'b> = &'b Foo;
    //~^ ERROR generic associated types are unstable [E0658]
    //~| ERROR lifetime parameters or bounds on type `Item` do not match the trait declaration

    fn next(&mut self) -> Option<Self::Item> {
        None
    }
}

fn main() { }
