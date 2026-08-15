#![feature(coverage_attribute)]
//@ edition: 2021

// Regression test for <https://github.com/rust-lang/rust/issues/122738>.
// These code patterns should not trigger an ICE when allocating a physical
// counter to a node and also one of its in-edges, because that is allowed
// when the node contains a tight loop to itself.

fn loopy(cond: bool) {
    let true = cond else { loop {} };
}

// Variant that also has `loop {}` on the success path.
// This isn't needed to catch the original ICE, but might help detect regressions.
fn _loop_either_way(cond: bool) {
    let true = cond else { loop {} };
    loop {}
}

// Variant using regular `if` instead of let-else.
// This doesn't trigger the original ICE, but might help detect regressions.
fn _if(cond: bool) {
    if cond { loop {} } else { loop {} }
}

#[coverage(off)]
fn main() {
    loopy(true);
}
