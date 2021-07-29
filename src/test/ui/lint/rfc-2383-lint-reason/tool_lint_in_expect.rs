// check-pass

#![feature(lint_reasons)]

#![expect(
    clippy::almost_swapped,
    reason = "This should be ignored in a normal run but trigger in a clippy run")]
fn main() {
    // See lint doc https://rust-lang.github.io/rust-clippy/master/index.html#almost_swapped
}
