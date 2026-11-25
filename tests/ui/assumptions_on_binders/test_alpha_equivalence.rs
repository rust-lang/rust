//@ check-pass
//@ compile-flags: -Zassumptions-on-binders -Znext-solver=globally

#![feature(test_binder_constraints)]

trait Trait { type Assoc; }

core::test_binder_constraints! {
    impl<T>
    where
        T: Trait,
        for<'x> T::Assoc: 'x,
    {
        forall {
            and {
                for<'a> T::Assoc: 'a,
                for<'b> T::Assoc: 'b,
            }
        } expect {
            for<'c> T::Assoc: 'c
        }
    }
}

fn main() {}
