// -*- rust -*-

// error-pattern:bounds check
fn main() {
    let s: ~str = ~"hello";

    // Bounds-check failure.
    assert (s[5] == 0x0 as u8);
}
