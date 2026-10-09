//@ check-pass
//@ compile-flags: -Zassumptions-on-binders -Znext-solver=globally

#![feature(test_binder_constraints, non_lifetime_binders)]
#![expect(incomplete_features)]

trait Trait<'a> {
    type Assoc;
}

core::test_binder_constraints! {
    impl<T>
    where
        for<'a> T: Trait<'a>,
        for<'a> <T as Trait<'a>>::Assoc: 'a
    {
        forall<'a, 'b>
        where
            'a: 'b,
        {
            for<> <T as Trait<'a>>::Assoc: 'b
        } expect {
            or {
                // this first candidate is the important one!
                for<'a> <T as Trait<'a>>::Assoc: 'a,
                // these won't wind up actually being used to prove the OR
                for<'a, 'b> <T as Trait<'a>>::Assoc: 'b,
                for<'a> <T as Trait<'a>>::Assoc: 'static,
            }
        }
    }
}

fn main() {}
