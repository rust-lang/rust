


// -*- rust -*-

// error-pattern:bounds check
fn main() {
    let str s = "hello";
    let int x = 0;
    assert (s.(x) == 0x68 as u8);
    // NB: at the moment a string always has a trailing NULL,
    // so the largest index value on the string above is 5, not
    // 4. Possibly change this.

    // Bounds-check failure.

    assert (s.(x + 6) == 0x0 as u8);
}