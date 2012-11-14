trait Foo {
    fn f();
}

struct Bar {
    x: int,
}

impl Bar : Drop {
    fn finalize() {}
}

impl Bar : Foo {
    fn f() {
        io::println("hi");
    }
}

fn main() {
    let x = ~Bar { x: 10 };
    let y = (move x) as ~Foo;   //~ ERROR uniquely-owned trait objects must be copyable
    let _z = copy y;
}

