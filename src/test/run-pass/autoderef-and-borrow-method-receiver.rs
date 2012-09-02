struct Foo {
    x: int;
}

impl Foo {
    fn f(&self) {}
}

fn g(x: &mut Foo) {
    x.f();
}

fn main() {
}

