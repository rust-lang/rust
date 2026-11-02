//@ check-pass

// test that we don't incorrectly try and eagerly handle alias outlives' from too
// low a universe

#![feature(test_binder_constraints, generic_const_items)]

trait AliasHaver {
    type Assoc;
}

core::test_binder_constraints! {
    impl<T: AliasHaver>
    where
        T::Assoc: 'static,
    {
        forall<'b> {
            for<> T::Assoc: 'static
        } expect {
            // previously this would just be `or {}` i.e. false
            for<> T::Assoc: 'static
        }
    }
}

fn main() {}
