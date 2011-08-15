// -*- rust -*-
type point = (int, int);

fn f(p: point, x: int, y: int) {
    let (a, b) = p;
    assert (a == x);
    assert (b == y);
}

fn main() {
    let p: point = (10, 20);
    let (a, b) = p;
    assert (a == 10);
    assert (b == 20);
    let p2: point = p;
    f(p, 10, 20);
    f(p2, 10, 20);
}