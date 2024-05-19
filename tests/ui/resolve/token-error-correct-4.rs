// Test that we do some basic error correction in the tokeniser and apply suggestions.

fn setsuna(_: ()) {}

fn kazusa() {}

fn main() {
    setsuna(kazusa();
} //~ ERROR mismatched closing delimiter
