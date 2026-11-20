//! It's an error to mutate a matched-on place in a guard, even if the guard unconditionally
//! diverges thereafter. See `divergent-guard-unconditional-tail-call.rs` for explicit tail calls.

fn main() {
    let mut x: Option<&u64> = Some(&7);
    let mut a: &[u8] = &[7];
    // The fake borrows keeping `x` and `a` from being mutated aren't live on the divergent paths
    // from the guards, so nothing is preventing the mutation.
    match a[0] {
        b if { a = &[]; return } => {}
        //~^ ERROR: cannot assign `a` in match guard
        _ => {}
    }
    loop {
        match x {
            Some(&y) if { x = None; return } => {}
            //~^ ERROR: cannot assign `x` in match guard
            Some(&y) if true || { x = None; return } => {}
            //~^ ERROR: cannot assign `x` in match guard
            Some(&y) if { x = None; panic!() } => {}
            //~^ ERROR: cannot assign `x` in match guard
            Some(&y) if { x = None; break } => {}
            //~^ ERROR: cannot assign `x` in match guard
            Some(&y) if { x = None; continue } => {}
            //~^ ERROR: cannot assign `x` in match guard

            _ => {},
        }
    }
}

fn discovered_use_case() -> Option<String> {
    // At the time of writing, creating a fake borrow temporary constrains not only the fake borrow
    // to be live at all points that the fake borrow temporary is; it applies that constraint to all
    // regions in covariant positions within the temporary's type. Unless that behavior is changed,
    // keeping fake borrow temporaries live on divergent paths can also keep real borrows live, as
    // shown by crater in <https://github.com/rust-lang/rust/pull/162242#issuecomment-5640060748>.
    let non_copy = None;
    match &non_copy {
        // This guard requires that the actual borrow of `non_copy` in the scrutinee outlives the
        // fake borrow temporary for the scrutinee place. Since the fake borrow temporary is live at
        // the end of the divergent path in the guard, the real borrow consequently is live there
        // too, preventing `non_copy` from being moved.
        Some(_) if return non_copy => None,
        //~^ ERROR: cannot move out of `non_copy` because it is borrowed

        _ => None,
    }
}
