// -*- rust -*-

// error-pattern:bounds check
// no-valgrind
fn main() {
    let v: [int] = ~[10];
    let x: int = 0;
    assert (v.(x) == 10);
    // Bounds-check failure.

    assert (v.(x + 2) == 20);
}