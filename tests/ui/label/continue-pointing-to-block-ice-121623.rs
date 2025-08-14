//! Regression test for ICE #121623. Liveness linting assumes that `continue`s all point to loops.
//! This tests that if a `continue` points to a block, we don't run liveness lints.

fn main() {
    match () {
        _ => 'b: {
            continue 'b;
            //~^ ERROR `continue` pointing to a labeled block
        }
    }
}
