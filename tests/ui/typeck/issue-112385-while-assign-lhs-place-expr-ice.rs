// Previously, the while loop with an assignment statement (mistakenly) as the condition
// which has a place expr as the LHS would trigger an ICE in typeck.
// Reduced from https://github.com/rust-lang/rust/issues/112385.

fn main() {
    let foo = Some(());
    while Some(foo) = None {}
    //~^ ERROR mismatched types
}
