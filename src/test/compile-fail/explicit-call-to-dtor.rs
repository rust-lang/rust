struct Foo {
    x: int
}

impl Foo : Drop {
    fn finalize() {
        io::println("kaboom");
    }
}

fn main() {
    let x = Foo { x: 3 };
    x.finalize();   //~ ERROR explicit call to destructor
}

