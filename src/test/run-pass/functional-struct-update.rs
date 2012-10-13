struct Foo {
    x: int,
    y: int
}

fn main() {
    let a = Foo { x: 1, y: 2 };
    let c = Foo { x: 4, .. a};
    io::println(fmt!("%?", c));
}

