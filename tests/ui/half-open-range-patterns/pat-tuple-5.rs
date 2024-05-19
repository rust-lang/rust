fn main() {
    const PAT: u8 = 1;

    match (0, 1) {
        (PAT ..) => {} //~ ERROR mismatched types
    }
}
