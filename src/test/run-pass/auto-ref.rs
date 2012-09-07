struct Foo {
    x: int,
}

trait Stuff {
    fn printme();
}

impl &Foo : Stuff {
    fn printme() {
        io::println(fmt!("%d", self.x));
    }
}

fn main() {
    let x = Foo { x: 3 };
    x.printme();
}

