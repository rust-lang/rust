enum Foo {
    Bar { bar: Bar, id: usize }
}

enum Bar {
    A, B, C, D, E, F
}

fn test(f: Foo) {
    match f {
        //~^ ERROR non-exhaustive patterns
        //~| patterns
        Foo::Bar { bar: Bar::A, .. } => (),
        Foo::Bar { bar: Bar::B, .. } => (),
    }
}

fn main() {}
