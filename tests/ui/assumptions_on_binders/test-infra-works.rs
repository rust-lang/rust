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

fn main() {}
