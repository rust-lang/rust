struct Foo {
    x: i32,
    y: i32
}

fn main() {
    let (x, y, z) = (0, 1, 2);
    let foo = Foo {
        x, y, z //~ ERROR struct `Foo` has no field named `z`
    };
}
