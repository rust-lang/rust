//@ compile-flags: -Znext-solver -Zassumptions-on-binders

// Regression test for #157778.
//
// A non-lifetime binder (`for<T>`) introduces a placeholder *type* in universe `u`. The
// resulting alias-outlives constraint `<!T as Trait>::Assoc: 'r` stays in `u` because the
// region-only rewrite (`PlaceholderReplacer` only folds regions) cannot pull a type
// placeholder out of `u`. This used to trip an `assert!(max_universe < u)` and ICE in
// `pull_region_outlives_constraints_out_of_universe`. The assumptions-on-binders machinery
// is region-outlives-only, so we now report ambiguity instead of panicking.

#![feature(non_lifetime_binders)]

trait Trait {
    type Assoc;
    type Ref //~ ERROR cannot satisfy `<T as Trait>::Assoc: 'static`
    where
        for<T> T: Trait<Assoc: 'static>;
}

fn main() {}
