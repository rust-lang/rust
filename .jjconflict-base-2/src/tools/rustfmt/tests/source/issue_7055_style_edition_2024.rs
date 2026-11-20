// rustfmt-style_edition: 2024

// The `removes_trailing_whitespace` case from the 2027 test is deliberately
// absent here: at style editions before 2027 rustfmt fails with "left behind
// trailing whitespace" and emits nothing, so there is no output to check.

struct S;

impl S {
    const fn new(_: &str) -> Self {
        S
    }

    fn go(&self) {}
}

fn main() {
    const {
        S::new(
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        )
    }
    .go();
}
