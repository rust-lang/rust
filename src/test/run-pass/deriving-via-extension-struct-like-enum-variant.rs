#[deriving_eq]
enum S {
    X { x: int, y: int },
    Y
}

pub fn main() {
    let x = X { x: 1, y: 2 };
    fail_unless!(x == x);
    fail_unless!(!(x != x));
}

