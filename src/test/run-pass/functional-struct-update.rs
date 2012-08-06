struct Foo {
    x: int;
    y: int;
}

fn main() {
    let a = Foo { x: 1, y: 2 };
    let b = Foo { x: 3 with a };
    let c = Foo { x: 4, with a };
    io::println(fmt!("%? %?", b, c));
}

