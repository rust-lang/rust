// -*- rust -*-

// error-pattern:index out of bounds: the len is 5 but the index is 5
fn main() {
    let s: ~str = ~"hello";

    // Bounds-check failure.
    assert (s[5] == 0x0 as u8);
}
