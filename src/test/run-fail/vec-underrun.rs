// -*- rust -*-

// error-pattern:index out of bounds: the len is 2 but the index is -1
fn main() {
    let v: ~[int] = ~[10, 20];
    let x: int = 0;
    assert (v[x] == 10);
    // Bounds-check failure.

    assert (v[x - 1] == 20);
}
