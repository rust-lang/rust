struct Foo {
    x: int
}

impl Foo : Drop {
    fn finalize(&self) {
        io::println("bye");
    }
}

fn main() {
    let x: Foo = Foo { x: 3 };
}

