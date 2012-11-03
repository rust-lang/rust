trait Foo {
    fn bar() -> ~str {
        fmt!("test")
    }
}

enum Baz {
    Quux
}

impl Baz: Foo {
}

fn main() {
    let q = Quux;
    assert q.bar() == ~"test";
}