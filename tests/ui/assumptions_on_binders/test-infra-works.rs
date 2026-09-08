//@ check-pass
//@ revisions: assumptions min_coroutines
//@[assumptions] compile-flags: -Zassumptions-on-binders
//@[min_coroutines] compile-flags: -Zassumptions-on-binders=min_coroutines

#![feature(test_binder_constraints, non_lifetime_binders)]
#![expect(incomplete_features)]

core::test_binder_constraints! {
    impl<'a: 'b, 'b> {
        'a: 'b
    }
}

core::test_binder_constraints! {
    impl<'a: 'b, 'b> {
        'a: 'b,
        forall { }
    }
}

// FIXME(-Zassumptions-on-binders): this should be `impl<'b, 'c: 'b>`, not
// `impl<'b, 'c: 'b + 'static>`, but OR isn't actually implemented yet
#[cfg(assumptions)]
core::test_binder_constraints! {
    impl<'b, 'c: 'b + 'static> {
        forall<'a> where 'b: 'a {
            'c: 'a
        } expect {
            or {
                'c: 'b,
                'c: 'static,
            }
        }
    }
}

core::test_binder_constraints! {
    impl<'a, T: 'a> {
        T: 'a,
        forall<T2> where T2: 'a {
            T2: 'a,
        }
    }
}

trait Trait {
    type Assoc;
}

// FIXME(-Zassumptions-on-binders): this probably shouldn't compile, the exit for the top-level
// `impl` should fail because the constraints asserted in `expect` should fail to prove true. Might
// be https://github.com/rust-lang/project-assumptions-on-binders/issues/26
//
// The `expect` clauses of this and the next test assert on the full mode's rewrite of alias
// outlives constraints into lower universes. `min_coroutines` deliberately does not rewrite, it
// only drops constraints directly implied by the binder's assumptions and keeps the rest as they
// are, so the rewritten form is specific to `assumptions`. The retained form cannot be spelled in
// an `expect` clause because it still mentions the binder's own lifetime, so `min_coroutines`
// coverage for aliases lives in `min-coroutines-alias-outlives.rs` instead.
//
// for<> syntax does direct insert into constraint storage
#[cfg(assumptions)]
core::test_binder_constraints! {
    impl<T: Trait> {
        forall<'a> {
            for<> T::Assoc: 'a
        } expect {
            or {
                for<'b> T::Assoc: 'b,
                for<> T::Assoc: 'static
            }
        }
    }
}

// FIXME(-Zassumptions-on-binders): this probably shouldn't compile, the exit for the top-level
// `impl` should fail because the constraints asserted in `expect` should fail to prove true. Might
// be https://github.com/rust-lang/project-assumptions-on-binders/issues/26
//
// `where` syntax goes through the full clause destructuring and register_obligation pipeline
#[cfg(assumptions)]
core::test_binder_constraints! {
    impl<T: Trait> {
        forall<'a> {
            where T::Assoc: 'a
        } expect {
            or {
                for<'b> T::Assoc: 'b,
                for<> T::Assoc: 'static,
                T: 'static
            }
        }
    }
}

#[cfg(min_coroutines)]
core::test_binder_constraints! {
    impl {
        // Minimal mode directly discharges constraints proven by the current binder without
        // rewriting either placeholder into a lower universe.
        forall<'a, 'b> where 'b: 'a {
            'b: 'a,
        } expect {}
    }
}

// Minimal mode discharges a type outlives goal when an assumption names the same type. Note that
// this case alone does not pin down whether the constraint was kept whole or destructured, since a
// bare param is its own only component either way; `min-coroutines-alias-outlives.rs` covers that.
#[cfg(min_coroutines)]
core::test_binder_constraints! {
    impl<T> {
        forall<'a> where T: 'a {
            where T: 'a
        } expect {}
    }
}

fn main() {}
