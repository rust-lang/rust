// -*- rust -*-

// error-pattern:index out of bounds: the len is 1 but the index is 2
fn main() {
    let v: ~[int] = ~[10];
    let x: int = 0;
    assert (v[x] == 10);
    // Bounds-check failure.

    assert (v[x + 2] == 20);
}
