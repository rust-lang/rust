//@ check-pass
//@ compile-flags: -Zassumptions-on-binders

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
// for<> syntax does direct insert into constraint storage
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

fn main() {}
