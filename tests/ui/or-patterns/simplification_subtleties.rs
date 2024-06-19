//@ run-pass

#[allow(unreachable_patterns)]
fn main() {
    // Test that we don't naively sort the two `2`s together and confuse the failure paths.
    match (1, true) {
        (1 | 2, false | false) => unreachable!(),
        (2, _) => unreachable!(),
        _ => {}
    }
}
