// Make sure unused parens lint emit is emitted for loop and match.
// See https://github.com/rust-lang/rust/issues/90807
// and https://github.com/rust-lang/rust/pull/91956#discussion_r771647953
#![deny(unused_parens)]

fn main() {
    for _ in (1..loop { break 2 }) {} //~ERROR
    for _ in (1..match () { () => 2 }) {} //~ERROR
}
