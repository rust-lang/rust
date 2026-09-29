// Regression test for <https://github.com/rust-lang/rust/issues/133947>.
//
// Make sure we don't ICE when there's `!` in a range pattern.
//
// Report the invalid runtime endpoint during typeck, before MIR building.

fn main() {
    let x: !;
    match 1 {
        0..x => {}
        //~^ ERROR runtime values cannot be referenced in patterns
    }
}
