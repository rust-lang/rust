//@ check-pass

fn repro() {
    trait Foo {
        type Bar;
    }

    #[allow(dead_code)]
    struct Baz<T: Foo> {
        field: T::Bar,
    }
}

fn main() {
    repro();
}
