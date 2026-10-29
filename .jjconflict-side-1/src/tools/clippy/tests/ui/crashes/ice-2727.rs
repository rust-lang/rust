//@ check-pass
// Test for https://github.com/rust-lang/rust-clippy/issues/2727

pub fn f(new: fn()) {
    new();
}

fn main() {}
