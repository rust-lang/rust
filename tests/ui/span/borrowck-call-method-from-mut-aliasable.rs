struct Foo {
    x: isize,
}

impl Foo {
    pub fn f(&self) {}
    pub fn h(&mut self) {}
}

fn a(x: &mut Foo) {
    x.f();
    x.h();
}

fn b(x: &Foo) {
    x.f();
    x.h(); //~ ERROR cannot borrow
}

fn main() {
}
