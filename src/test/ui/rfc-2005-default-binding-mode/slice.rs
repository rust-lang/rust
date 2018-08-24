#![feature(slice_patterns)]

pub fn main() {
    let sl: &[u8] = b"foo";

    match sl { //~ ERROR non-exhaustive patterns
        [first, remainder..] => {},
    };
}
