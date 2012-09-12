struct Foo {
    x: int,
}

impl Foo {
    fn f(&const self) {}
}

fn g(x: &mut Foo) {
    x.f();
}

fn main() {
}

