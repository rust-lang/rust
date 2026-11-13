//! On a guard's success path, we add fake-reads for fake-borrows on prefixes of by-value bindings'
//! places, since at that point we need to be able to create by-value bindings. For cases where we
//! omit unnecessary fake reads on the success path, see `divergent-guard-binding-permissivenss.rs`.

fn main() {
    let mut x: Option<&u64> = Some(&7);
    match x {
        // These guards can succeed but not fail after `x = None`. Since `x` is a prefix of the by-
        // value binding `b`'s place, we can't bind `b` if we do that.
        Some(&b) if { x = None; false } || return => {}
        //~^ ERROR: cannot assign `x` in match guard
        Some(&b) if false || ({ x = None; false } || return) => {}
        //~^ ERROR: cannot assign `x` in match guard

        // If there isn't a deref between the modified place and the bound place, this may be caught
        // instead by the fake read on the `RefWithinGuard` borrow of the binding.
        Some(b) if { x = None; false } || return => {}
        //~^ ERROR: cannot assign to `x` because it is borrowed
        Some(b) if false || ({ x = None; false } || return) => {}
        //~^ ERROR: cannot assign to `x` because it is borrowed

        // By-ref bindings are created before evaluating the guard, so we don't need fake borrows
        // to catch when they would be invalidated.
        Some(ref b) if { x = None; false } || return => { b; }
        //~^ ERROR: cannot assign to `x` because it is borrowed
        Some(ref b) if false || ({ x = None; false } || return) => { b; }
        //~^ ERROR: cannot assign to `x` because it is borrowed

        _ => {}
    }
}
