trait Show {
    #[derivable]
    fn show(&self);
}

impl int : Show {
    fn show(&self) {
        io::println(self.to_str());
    }
}

struct Foo {
    x: int,
    y: int,
    z: int,
}

impl Foo : Show;

enum Bar {
    Baz(int, int),
    Boo(Foo),
}

impl Bar : Show;

fn main() {
    let foo = Foo { x: 1, y: 2, z: 3 };
    foo.show();

    io::println("---");

    let baz = Baz(4, 5);
    baz.show();

    io::println("---");

    let boo = Boo(Foo { x: 6, y: 7, z: 8 });
    boo.show();
}

