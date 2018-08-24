#[derive(Debug)]
struct foo {
    x: isize
}

fn main() {
    println!("{}", foo{ x: 1 } as isize); //~ non-primitive cast: `foo` as `isize` [E0605]
}
