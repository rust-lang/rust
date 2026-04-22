//! Regression test for https://github.com/rust-lang/rust/issues/3154

struct Thing<'a, Q:'a> {
    x: &'a Q
}

fn thing<'a,Q>(x: &Q) -> Thing<'a,Q> {
    Thing { x: x } //~ ERROR explicit lifetime required in the type of `x` [E0621]
}

fn main() {
    thing(&());
}
