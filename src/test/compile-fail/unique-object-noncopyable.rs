trait Foo {
    fn f();
}

struct Bar {
    x: int,
    drop {}
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

