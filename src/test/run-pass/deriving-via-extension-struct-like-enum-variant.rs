#[deriving(Eq)]
enum S {
    X { x: int, y: int },
    Y
}

pub fn main() {
    let x = X { x: 1, y: 2 };
    assert_eq!(x, x);
    assert!(!(x != x));
}
