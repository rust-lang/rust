struct Foo {
    x: int
}

trait Bar : Drop {
    fn blah();
}

impl Foo : Drop {
    fn finalize(&self) {
        io::println("kaboom");
    }
}

impl Foo : Bar {
    fn blah() {
        self.finalize();    //~ ERROR explicit call to destructor
    }
}

fn main() {
    let x = Foo { x: 3 };
}


