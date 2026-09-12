//@ check-pass
//@ compile-flags: -Zassumptions-on-binders

#![feature(test_binder_constraints, non_lifetime_binders)]
#![expect(incomplete_features)]

// Root type outlives constraints are destructured into an OR over every region the type is known
// to outlive. The reflexive `'b: 'b` candidate makes this OR true even though the unrelated
// `'a: 'b` candidate does not hold. Reflexive leaves are dropped when building an AND, which
// leaves an empty, i.e. trivially true, AND as one of the candidates of the OR.
core::test_binder_constraints! {
    impl<'a, 'b, T: 'a + 'b> {
        T: 'b
    }
}

fn main() {}
