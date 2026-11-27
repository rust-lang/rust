//! Companion tests to `divergent-guard-bindings.rs`: it's fine to mutate a matched-on place in a
//! guard as long as it can't fail after mutation and no sub-places of a matched-on place are bound
//! by value.
//@ check-pass

fn main() {
    let mut x: (Option<&Box<u64>>, u64) = (Some(&Box::new(7)), 0);
    match x {
        // We match on `x.0`.
        (None, _) => {}

        // These arms can't continue matching after assigning to `x.0`; the guard either succeeds or
        // diverges after that point.
        (_, c) if { x.0 = None; false } || return => {}
        (_, c) if false && ({ x.0 = None; false } || return) => {}

        // Unlike by-value bindings, by-ref bindings are created before evaluating the guard. The
        // borrow-checker would raise a conflict error if `b` was used after writing to `x.0` (see
        // `divergent-guard-bindings.rs`).
        (Some(ref b), _) if { x.0 = None; false } || return => {}
        (Some(ref b), _) if false && ({ x.0 = None; false } || return) => {}

        // This assignment doesn't invalidate borrows of the lifetime-extended temporary box. It's
        // unchanged even when `x` is set to `None`.
        (Some(&ref b), c) if { x.0 = None; false } || return => { **b; }
        (Some(&ref b), c) if false && ({ x.0 = None; false } || return) => { **b; }

        a => { a; }
    }

    let mut s: &[u8] = &[7];
    match s[0] {
        // No by-value binding is created and the guard can't fail.
        _ if { s = &[]; false } || return => {}
        _ if false || ({ s = &[]; false } || return) => {}

        // Reassigning `s` doesn't affect `b`, which still borrows from the original slice.
        ref b if { s = &[]; false } || return => { *b; }
        ref b if false || ({ s = &[]; false } || return) => { *b; }

        _ => {}
    }
}
