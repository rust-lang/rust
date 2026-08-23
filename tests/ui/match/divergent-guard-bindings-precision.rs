//! Companion tests to `divergent-guard-bindings.rs`: it's an error to mutate a matched-on place in
//! a guard that can succeed, even if it can't fail and no sub-places of the mutated place are bound
//! by value.

fn main() {
    let mut x: (Option<&Box<u64>>, u64) = (Some(&Box::new(7)), 0);
    match x {
        // We match on `x.0`.
        (None, _) => {}

        // These arms can't continue matching after assigning to `x.0`; the guard either succeeds or
        // diverges after that point. Technically it would be sound to permit these.
        (_, c) if { x.0 = None; false } || return => {}
        //~^ ERROR: cannot assign `x.0` in match guard
        (_, c) if false && ({ x.0 = None; false } || return) => {}
        //~^ ERROR: cannot assign `x.0` in match guard

        // Unlike by-value bindings, by-ref bindings are created before evaluating the guard. Since
        // `b` isn't used after its borrow is invalidated, this technically would be sound.
        (Some(ref b), _) if { x.0 = None; false } || return => {}
        //~^ ERROR: cannot assign `x.0` in match guard
        (Some(ref b), _) if false && ({ x.0 = None; false } || return) => {}
        //~^ ERROR: cannot assign `x.0` in match guard

        // This assignment doesn't invalidate borrows of the lifetime-extended temporary box. It's
        // unchanged even when `x` is set to `None`. As above, this would be sound.
        (Some(&ref b), c) if { x.0 = None; false } || return => { **b; }
        //~^ ERROR: cannot assign `x.0` in match guard
        (Some(&ref b), c) if false && ({ x.0 = None; false } || return) => { **b; }
        //~^ ERROR: cannot assign `x.0` in match guard

        a => { a; }
    }

    let mut s: &[u8] = &[7];
    match s[0] {
        // No binding is created and the guard can't fail, so technically it would be sound to
        // permit this.
        _ if { s = &[]; false } || return => {}
        //~^ ERROR: cannot assign `s` in match guard
        _ if false || ({ s = &[]; false } || return) => {}
        //~^ ERROR: cannot assign `s` in match guard

        // Likewise, this could technically be fine. The guard can't fail and reassigning `s`
        // doesn't affect `b`, which still borrows from the original slice.
        ref b if { s = &[]; false } || return => { *b; }
        //~^ ERROR: cannot assign `s` in match guard
        ref b if false || ({ s = &[]; false } || return) => { *b; }
        //~^ ERROR: cannot assign `s` in match guard

        _ => {}
    }
}
