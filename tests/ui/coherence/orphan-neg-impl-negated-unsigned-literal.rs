// Regression test for <https://github.com/rust-lang/rust/issues/165689>.
// An orphan `impl Neg for u8` let `-1: u8` reach MIR building, which ICEd on the negated literal.

impl std::ops::Neg for u8 {
    //~^ ERROR only traits defined in the current crate can be implemented for primitive types
    type Output = u8;
    fn neg(self) -> Self::Output {
        -1
    }
}

fn main() {}
