struct Foo {
    x: int,
    y: int,
}

fn main() {
    let a = Foo { x: 1, y: 2 };
    match a {
        Foo { x: x, y: y } => io::println(fmt!("yes, %d, %d", x, y))
    }
}



