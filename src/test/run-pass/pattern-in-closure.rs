struct Foo {
    x: int,
    y: int
}

fn main() {
    let f = |(x, _): (int, int)| io::println((x + 1).to_str());
    let g = |Foo { x: x, y: y }: Foo| io::println((x + 1).to_str());
    f((2, 3));
    g(Foo { x: 1, y: 2 });
}

