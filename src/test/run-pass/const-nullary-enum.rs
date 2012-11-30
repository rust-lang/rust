enum Foo {
    Bar,
    Baz,
    Boo,
}

const X: Foo = Bar;

fn main() {
    match X {
        Bar => {}
        Baz | Boo => fail
    }
}

