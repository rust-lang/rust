#[deriving_eq]
enum S {
    X { x: int, y: int },
    Y
}

fn main() {
    let x = X { x: 1, y: 2 };
    assert x == x;
    assert !(x != x);
}

