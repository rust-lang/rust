//@ check-pass
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver=globally

// Check that elaborating a higher-ranked supertrait correctly shifts
// bound-variable indices when combining the two sets of binders.

trait Triple<'a, 'b, 'c> {}

trait Double<'a, 'b>: for<'c> Triple<'a, 'b, 'c> {}

fn needs_triple<T: for<'a, 'b, 'c> Triple<'a, 'b, 'c>>() {}

fn implied_supertrait<T: for<'a, 'b> Double<'a, 'b>>() {
    needs_triple::<T>();
}

fn main() {}
