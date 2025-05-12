// Regression test for <https://github.com/rust-lang/rust/issues/133947>.

// Make sure we don't ICE when there's `!` in a range pattern.
//
// This shouldn't be allowed anyways, but we only deny it during MIR
// building, so make sure we handle it semi-gracefully during typeck.

#![feature(never_type)]

fn main() {
    let x: !;
    match 1 {
        0..x => {}
        //~^ ERROR only `char` and numeric types are allowed in range patterns
    }
}
