#![feature(destructuring_assignment)]
struct Struct<S, T> {
    a: S,
    b: T,
}

fn main() {
    let (mut a, b);
    let mut c;
    Struct { a, b, c } = Struct { a: 0, b: 1 }; //~ ERROR does not have a field named `c`
}
