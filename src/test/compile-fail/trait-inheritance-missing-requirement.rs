// xfail-test
// error-pattern: what

trait Foo {
    fn f();
}

trait Bar : Foo {
    fn g();
}

struct A {
    x: int
}

// Can't implement Bar without an impl of Foo
impl A : Bar {
    fn g() { }
}

fn main() {
}

