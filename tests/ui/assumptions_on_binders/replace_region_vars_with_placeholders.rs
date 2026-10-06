//@ compile-flags: -Zassumptions-on-binders -Znext-solver=globally
//@ check-pass

// test that when rewriting alias outlives constraints we will replace
// region variables in the universe being handled, with (new) bound vars
// on the alias outlives constraint's binder.
//
// we previously did not do this and would get ambiguity which was overly
// conservative

#![feature(test_binder_constraints)]

trait Trait<'a> {
    type Assoc;
}

core::test_binder_constraints! {
    impl<T: for<'a> Trait<'a>> {
        forall<'b>
        where
            for<'a> <T as Trait<'a>>::Assoc: 'b
        {
            forall<> {
                exists<'a> {
                    for<> <T as Trait<'a>>::Assoc: 'b
                }
            } expect {
                for<'a> <T as Trait<'a>>::Assoc: 'b
            }
        } expect {
            or { and {} }
        }
    }
}

fn main() {}
