#[derive(Debug)]
struct Foo {
    x: isize
}

fn main() {
    println!("{}", Foo { x: 1 } as isize); //~ non-primitive cast: `Foo` as `isize` [E0605]
}
