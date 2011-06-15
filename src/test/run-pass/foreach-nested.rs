


// -*- rust -*-
iter two() -> int { put 0; put 1; }

fn main() {
    let vec[mutable int] a = [mutable -1, -1, -1, -1];
    let int p = 0;
    for each (int i in two()) {
        for each (int j in two()) { a.(p) = 10 * i + j; p += 1; }
    }
    assert (a.(0) == 0);
    assert (a.(1) == 1);
    assert (a.(2) == 10);
    assert (a.(3) == 11);
}