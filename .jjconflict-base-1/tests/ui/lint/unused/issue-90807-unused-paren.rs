//@ check-pass
// Make sure unused parens lint doesn't emit a false positive.
// See https://github.com/rust-lang/rust/issues/90807
#![deny(unused_parens)]

fn main() {
    for _ in (1..{ 2 }) {}
}
