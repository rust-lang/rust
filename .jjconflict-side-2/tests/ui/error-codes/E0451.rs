mod bar {
    pub struct Foo {
        pub a: isize,
        b: isize,
    }

    pub struct FooTuple (
        pub isize,
        isize,
    );
}

fn pat_match(foo: bar::Foo) {
    let bar::Foo{a, b} = foo; //~ ERROR E0451
}

fn main() {
    let f = bar::Foo{ a: 0, b: 0 }; //~ ERROR E0451
}
