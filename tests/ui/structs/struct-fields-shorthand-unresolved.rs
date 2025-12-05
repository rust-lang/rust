struct Foo {
    x: i32,
    y: i32
}

fn main() {
    let x = 0;
    let foo = Foo {
        x,
        y //~ ERROR cannot find value `y` in this scope
    };
}
