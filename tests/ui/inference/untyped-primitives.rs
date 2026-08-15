//@ check-pass
// issue: rust-lang/rust#123824
// This test is a sanity check and does not enforce any stable API, so may be
// removed at a future point.

fn main() {
    let x = f32::from(3.14);
    //~^ WARN falling back to `f32`
    //~| WARN this was previously accepted
    let y = f64::from(3.14);
}
