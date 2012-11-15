trait Trait {
    #[derivable]
    fn f(&self, x: int, y: &str);
}

impl int : Trait {
    fn f(&self, x: int, y: &str) {
        assert x == 42;
        assert y == "hello";
    }
}

impl float : Trait {
    fn f(&self, x: int, y: &str) {
        assert x == 42;
        assert y == "hello";
    }
}

struct Foo {
    x: int,
    y: float
}

impl Foo : Trait;

fn main() {
    let a: Foo = Foo { x: 1, y: 2.0 };
    a.f(42, "hello");
}

