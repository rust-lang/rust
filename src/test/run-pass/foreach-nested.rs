


// -*- rust -*-
iter two() -> int { put 0; put 1; }

fn main() {
    let a: vec[mutable int] = [mutable -1, -1, -1, -1];
    let p: int = 0;
    for each i: int  in two() {
        for each j: int  in two() { a.(p) = 10 * i + j; p += 1; }
    }
    assert (a.(0) == 0);
    assert (a.(1) == 1);
    assert (a.(2) == 10);
    assert (a.(3) == 11);
}