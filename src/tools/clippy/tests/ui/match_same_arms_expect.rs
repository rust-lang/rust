//! An `#[expect(clippy::match_same_arms)]` on a match arm must still work when the lint is
//! allowed on an enclosing scope.
//@check-pass
#![allow(clippy::match_same_arms)]

fn allowed_outer_expect_inner(x: u32) -> u32 {
    match x {
        #[expect(clippy::match_same_arms)]
        1 => 42,
        #[expect(clippy::match_same_arms)]
        2 => 42,
        _ => 0,
    }
}

fn main() {
    allowed_outer_expect_inner(1);
}
