struct Foo {
    x: int,
}

impl Foo {
    fn f(&self) {}
    fn g(&const self) {}
    fn h(&mut self) {}
}

fn a(x: &mut Foo) {
    x.f(); //~ ERROR illegal borrow unless pure
    x.g();
    x.h();
}

fn b(x: &Foo) {
    x.f();
    x.g();
    x.h(); //~ ERROR illegal borrow
}

fn c(x: &const Foo) {
    x.f(); //~ ERROR illegal borrow unless pure
    x.g();
    x.h(); //~ ERROR illegal borrow
}

fn main() {
}

