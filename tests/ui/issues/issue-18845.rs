//@ run-pass
// This used to generate invalid IR in that even if we took the
// `false` branch we'd still try to free the Box from the other
// arm. This was due to treating `*Box::new(9)` as an rvalue datum
// instead of as a place.

fn test(foo: bool) -> u8 {
    match foo {
        true => *Box::new(9),
        false => 0
    }
}

fn main() {
    assert_eq!(9, test(true));
}
