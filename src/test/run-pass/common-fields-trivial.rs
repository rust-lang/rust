enum Foo {
    struct {
        x: int,
        y: int,
    }

    Bar(int),
    Baz(int)
}

fn main() {
    let x = Bar(3);
}

