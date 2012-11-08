struct Foo {
    x: int,
    y: int
}

fn main() {
    let f = |(x, _): (int, int)| assert x == 2;
    let g = |Foo { x: x, y: y }: Foo| {
        assert x == 1;
        assert y == 2;
    };
    f((2, 3));
    g(Foo { x: 1, y: 2 });
}

