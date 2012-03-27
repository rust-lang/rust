


// -*- rust -*-
type point = {x: int, y: int, mut z: int};

fn f(&p: point) { p.z = 13; }

fn main() {
    let mut x: point = {x: 10, y: 11, mut z: 12};
    f(x);
    assert (x.z == 13);
}
