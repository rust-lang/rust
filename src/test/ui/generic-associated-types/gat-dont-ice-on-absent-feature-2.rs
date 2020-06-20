// rust-lang/rust#60654: Do not ICE on an attempt to use GATs that is
// missing the feature gate.

struct Foo;

trait MyTrait {
    type Item<T>;
    //~^ ERROR generic associated types are unstable [E0658]
}

impl MyTrait for Foo {
    type Item<T> = T;
    //~^ ERROR generic associated types are unstable [E0658]
}

fn main() { }
