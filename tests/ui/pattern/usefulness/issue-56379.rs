enum Foo {
    A(bool),
    B(bool),
    C(bool),
}

fn main() {
    match Foo::A(true) {
        //~^ ERROR non-exhaustive patterns: `Foo::A(false)`, `Foo::B(false)` and `Foo::C(false)` not covered
        Foo::A(true) => {}
        Foo::B(true) => {}
        Foo::C(true) => {}
    }
}
