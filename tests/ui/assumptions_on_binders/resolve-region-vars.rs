//@ check-pass
//@ compile-flags: -Zassumptions-on-binders -Znext-solver=globally

#![feature(test_binder_constraints)]

trait Trait<'a, 'b> {
    type Assoc;
}

core::test_binder_constraints! {
    impl<'c, T>
    where
        for<'a, 'b> T: Trait<'a, 'b>,
    {
        forall<'a>
        where
            for<'b> <T as Trait<'a, 'b>>::Assoc: 'c,
        {
            forall<'b> {
                predicates <T as Trait<'a, 'b>>::Assoc: 'c
            } expect {
                for<'b> <T as Trait<'a, 'b>>::Assoc: 'c
            }
        } expect { }
    }
}

core::test_binder_constraints! {
    impl<'c, T>
    where
        for<'a, 'b> T: Trait<'a, 'b>,
    {
        forall<'a>
        where
            for<'b> <T as Trait<'a, 'b>>::Assoc: 'c,
        {
            forall<'b> {
                exists<'a2, 'b2, 'c2> {
                    'a2 = 'a,
                    'b2 = 'b,
                    'c2 = 'c,
                    or {
                        for<> <T as Trait<'a2, 'b2>>::Assoc: 'c2,
                        and { T: 'c2, 'a2: 'c2, 'b2: 'c2 }
                    }
                }
            } expect {
                for<'b> <T as Trait<'a, 'b>>::Assoc: 'c
            }
        } expect { }
    }
}

core::test_binder_constraints! {
    impl<'c, T>
    where
        for<'a, 'b> T: Trait<'a, 'b>,
    {
        forall<'a>
        where
            for<'b> <T as Trait<'a, 'b>>::Assoc: 'c,
        {
            forall<'b> {
                exists<'a2, 'b2, 'c2> {
                    exists<'a3, 'b3, 'c3> {
                        'a2 = 'a3,
                        'b2 = 'b3,
                        'c2 = 'c3,
                        'a3 = 'a,
                        'b3 = 'b,
                        'c3 = 'c,
                        for<> <T as Trait<'a3, 'b3>>::Assoc: 'c3
                    }
                }
            } expect {
                for<'b> <T as Trait<'a, 'b>>::Assoc: 'c
            }
        } expect { }
    }
}

fn main() {}
