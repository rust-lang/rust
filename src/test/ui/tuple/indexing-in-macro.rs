// Broken by https://github.com/rust-lang/rust/pull/70420.

macro_rules! m {
    (.$l:literal) => {};
}

m!(.0.0); //~ ERROR no rules expected the token `.`

fn main() {}
