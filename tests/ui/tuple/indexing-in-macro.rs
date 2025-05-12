//@ check-pass

macro_rules! m {
    (.$l:literal) => {};
}

m!(.0.0); // OK, `0.0` after a dot is still a float token.

fn main() {}
