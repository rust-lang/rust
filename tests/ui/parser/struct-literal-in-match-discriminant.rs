struct Foo {
    x: isize,
}

fn main() {
    match Foo { //~ ERROR struct literals are not allowed here
        x: 3
    } {
        Foo {
            x: x
        } => {}
    }
}
