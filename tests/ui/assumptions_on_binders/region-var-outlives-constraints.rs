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
        //~^ ERROR unable to satisfy constraints involving placeholders due to unknown implied bounds
        where
            for<'b> <T as Trait<'a, 'b>>::Assoc: 'c,
        {
            forall<'b> {
                exists<'a2, 'b2, 'c2> {
                    'a2: 'a, 'a: 'a2,
                    'b2: 'b, 'b: 'b2,
                    'c2: 'c, 'c: 'c2,
                    predicates <T as Trait<'a2, 'b2>>::Assoc: 'c2
                }
            } expect { ambiguity }
        } expect { ambiguity }
    }
}

fn main() {}
