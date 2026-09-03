//! Regression test for <https://github.com/rust-lang/rust/issues/161852>: we need to keep fake
//! borrows on indexed-into slice pointers alive for bounds-checks even if the end of the indexing
//! chain is unreachable. This prevents bounds-checks from performing out-of-bounds accesses.
//@ check-pass
// TODO: this should be check-fail

#![feature(explicit_tail_calls)]

fn main() {
    let mut x: &[&[&[u32]]] = &[&[&[0]]];
    let y: &[&[&[u32]]] = &[];
    x[0][{ x = y; 0 }][{ return; 0 }];
    // TODO: ERROR: cannot assign `x` in indexing expression
}

// In the following tests, no bounds-checks are reachable after mutating `x`. We keep the fake
// borrow on it alive for consistency, though it isn't necessary for soundness.

fn always_return_after_mutation() {
    let mut x: &[&[&[u32]]] = &[&[&[0]]];
    let y: &[&[&[u32]]] = &[];
    x[0][{ x = y; return; 0 }];
    // TODO: ERROR: cannot assign `x` in indexing expression
}

fn always_panic_after_mutation() {
    let mut x: &[&[&[u32]]] = &[&[&[0]]];
    let y: &[&[&[u32]]] = &[];
    x[0][{ x = y; panic!(); 0 }];
    // TODO: ERROR: cannot assign `x` in indexing expression
}

fn always_break_after_mutation() {
    let mut x: &[&[&[u32]]] = &[&[&[0]]];
    let y: &[&[&[u32]]] = &[];
    'b: {
        x[0][{ x = y; break 'b; 0 }];
        // TODO: ERROR: cannot assign `x` in indexing expression
    }
}

fn always_continue_after_mutation() {
    let mut x: &[&[&[u32]]] = &[&[&[0]]];
    let y: &[&[&[u32]]] = &[];
    loop {
        x[0][{ x = y; continue; 0 }];
        // TODO: ERROR: cannot assign `x` in indexing expression
    }
}

fn always_become_after_mutation() {
    let mut x: &[&[&[u32]]] = &[&[&[0]]];
    let y: &[&[&[u32]]] = &[];
    x[0][{ x = y; become always_become_after_mutation(); 0 }];
    // TODO: ERROR: cannot assign `x` in indexing expression
}
