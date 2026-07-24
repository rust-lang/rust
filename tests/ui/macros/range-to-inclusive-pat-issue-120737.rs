//@ check-pass

// Ensure macro `:pat` fragments accept `..=` as a valid pattern start.

fn main() {
    assert!(matches!(2, ..=0 | 2));
    assert!(matches!(0, ..=0));
}
