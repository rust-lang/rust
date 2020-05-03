// compile-flags: -Z mir-opt-level=3
// check-pass

// This test is designed to never trigger `unconditional_panic`.
// If it does, there's a bug in the MIR optimizations.
#![deny(unconditional_panic)]
fn main() {
    let z = combine_wisely(1, 0);
}

// If `combine_wisely` is called with a `y` value
// equal to zero, the branch where we compute `x/y`
// will not be executed. Our constant propagation pass must
// be smart enough to avoid computing these invalid states
// if they are, semantically, never going to happen.
//
// This test will fail if we ever drop the ball with
// regards to constant propagation and its savvyness
// concerning control flow.
#[inline(never)]
fn combine_wisely(x: u32, y: u32) -> u32 {
    if y != 0 {
        x / y
    } else {
        x + y
    }
}
