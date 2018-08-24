mod Bar {
    pub struct Foo {
        pub a: isize,
        b: isize,
    }

    pub struct FooTuple (
        pub isize,
        isize,
    );
}

fn pat_match(foo: Bar::Foo) {
    let Bar::Foo{a:a, b:b} = foo; //~ ERROR E0451
}

fn main() {
    let f = Bar::Foo{ a: 0, b: 0 }; //~ ERROR E0451
}
