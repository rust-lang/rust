trait Foo {
    fn f() {
        io::println("Hello!");
        self.g();
    }
    fn g();
}

struct A {
    x: int
}

impl A : Foo {
    fn g() {
        io::println("Goodbye!");
    }
}

fn main() {
    let a = A { x: 1 };
    a.f();
}

