trait Foo {
    fn f();
}

struct Bar {
    x: int
}

impl Bar : Foo {
    fn f() {
        io::println("hi");
    }
}

fn main() {
    let x = ~Bar { x: 10 };
    let y = x as ~Foo;
    let z = copy y;
    y.f();
    z.f();
}

