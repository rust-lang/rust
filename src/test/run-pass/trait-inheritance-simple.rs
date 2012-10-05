trait Foo {
    fn f();
}

trait Bar : Foo {
    fn g();
}

struct A {
    x: int
}

impl A : Bar {
    fn g() { io::println("in g"); }
    fn f() { io::println("in f"); }
}

fn h<T:Foo>(a: &T) {
    a.f();
}

fn main() {
    let a = A { x: 3 };
    h(&a);
}

