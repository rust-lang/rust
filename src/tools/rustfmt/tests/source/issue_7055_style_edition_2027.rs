// rustfmt-style_edition: 2027

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

fn removes_trailing_whitespace() {
    const {
        S::new( 
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        )
    }
    .go();
}
