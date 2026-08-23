//! Test that even if a match guard can't fail after mutating the scrutinee, it can't bind sub-
//! places of the mutated place if the arm body is reachable.

fn main() {
    let mut x: Option<&u64> = Some(&7);
    match x {
        // These guards can succeed but not fail after `x = None`. We can't bind `b` if we do that.
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

    let mut a: &[u8] = &[7];
    match a[0] {
        // These guards can succeed but not fail after `a = &[]`. We can't bind `b` if we do that.
        b if { a = &[]; false } || return => {}
        //~^ ERROR: cannot assign `a` in match guard
        b if false || ({ a = &[]; false } || return) => {}
        //~^ ERROR: cannot assign `a` in match guard

        _ => {}
    }
}
