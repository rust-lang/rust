enum Foo {
    A(bool),
    B(bool),
    C(bool),
}

fn main() {
    match Foo::A(true) {
        //~^ ERROR match is non-exhaustive
        Foo::A(true) => {}
        Foo::B(true) => {}
        Foo::C(true) => {}
    }
}
