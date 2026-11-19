//@ check-pass
//@ compile-flags: -Zassumptions-on-binders -Znext-solver=globally

#![feature(test_binder_constraints, non_lifetime_binders)]
#![expect(incomplete_features)]

trait Same<T> {}
impl<T> Same<T> for T {}

trait Trait<'a, 'b> {
    type Assoc;
}

struct Carrier<T>(T);
trait Require<'a, 'b, 'c, T> {}

impl<'a, 'b, 'c, T> Require<'a, 'b, 'c, T> for Carrier<T>
where
    T: Trait<'a, 'b>,
    <T as Trait<'a, 'b>>::Assoc: 'c,
{
}

core::test_binder_constraints! {
    impl<'a: 'b, 'b: 'a> {
        forall<> {
            predicates &'a (): Same<&'b ()>
        } expect {
            'b: 'a,
            'a: 'b,
        }
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
                predicates Carrier<T>: Require<'a, 'b, 'c, T>
            } expect {
                for<'b> <T as Trait<'a, 'b>>::Assoc: 'c
            }
        } expect {}
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
                exists<X> {
                    predicates Carrier<X>: Require<'a, 'b, 'c, T>
                }
            } expect {
                for<'b> <T as Trait<'a, 'b>>::Assoc: 'c
            }
        } expect {}
    }
}

fn main() {}
