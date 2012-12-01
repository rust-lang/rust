struct Foo;

const X: Foo = Foo;

fn main() {
    match X {
        Foo => {}
    }
}

