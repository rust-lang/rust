pub fn main() {
    let sl: &[u8] = b"foo";

    match sl { //~ ERROR match is non-exhaustive
        [first, remainder @ ..] => {},
    };
}
