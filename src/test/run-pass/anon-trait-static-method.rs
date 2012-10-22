struct Foo {
    x: int
}

impl Foo {
    static fn new() -> Foo {
        Foo { x: 3 }
    }
}

fn main() {
    let x = Foo::new();
    io::println(x.x.to_str());
}

