//@ check-pass

#![expect(clippy::no_effect)]

// Test for https://github.com/rust-lang/rust-clippy/issues/1588

fn main() {
    match 1 {
        1 => {},
        2 => {
            [0; 1];
        },
        _ => {},
    }
}
