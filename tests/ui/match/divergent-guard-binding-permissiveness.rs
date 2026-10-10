//! On a guard's success path, we only add fake-reads for fake borrows on prefixes of by-value
//! bindings' places, since at that point we only need to be able to create by-value bindings.
//! Consequently, guards that can't fail after a matched-on place has been modified may still
//! borrow-check successfully if no by-value bindings depended on that place to still be valid. See
//! `divergent-guard-binding-restrictiveness.rs` for cases where the presence of bindings causes a
//! borrow-checking failure.
//@ check-pass
#![allow(unused)]

fn main() {
    let mut x: (Option<&Box<u64>>, u64) = (Some(&Box::new(7)), 0);
    match x {
        // By-ref bindings are created before evaluating the guard, so we don't need fake borrows
        // to catch when they would be invalidated. As long as we don't use `b` after, it's ok.
        (Some(ref b), c) if { x.0 = None; false } || return => {}
        (Some(ref b), c) if false && ({ x.0 = None; false } || return) => {}

        // The assignment doesn't invalidate borrows of the lifetime-extended temporary box. It's
        // unchanged even when `x` is set to `None`, so it's fine to read from after.
        (Some(&ref b), c) if { x.0 = None; false } || return => { **b; }
        (Some(&ref b), c) if false && ({ x.0 = None; false } || return) => { **b; }

        a => { a; }
    }
}
