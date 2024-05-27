//@ known-bug: rust-lang/rust#125323
fn main() {
    for _ in 0..0 {
        [(); loop {}];
    }
}
