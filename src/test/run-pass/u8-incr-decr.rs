


// -*- rust -*-

// These constants were chosen because they aren't used anywhere
// in the rest of the generated code so they're easily grep-able.
fn main() {
    let x: u8 = 19u8; // 0x13

    let y: u8 = 35u8; // 0x23

    x = x + 7u8; // 0x7

    y = y - 9u8; // 0x9

    assert (x == y);
}