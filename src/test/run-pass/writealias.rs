


// -*- rust -*-
type point = {x: int, y: int, mutable z: int};

fn f(p: &mutable point) { p.z = 13; }

fn main() {
    let x: point = {x: 10, y: 11, mutable z: 12};
    f(x);
    assert (x.z == 13);
}