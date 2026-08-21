//@ compile-flags: -Zassumptions-on-binders
#![feature(test_binder_constraints)]
#![expect(incomplete_features)]

core::test_binder_constraints! {
    //~^ ERROR unsatisfied lifetime constraint from -Zassumptions-on-binders :3
    impl<'a, 'b> {
        'a: 'b
    }
}

core::test_binder_constraints! {
    impl {
        and {
            forall { }
            //~^ ERROR expected one of
        }
    }
}

core::test_binder_constraints! {
    impl {
        or {
            forall { }
            //~^ ERROR expected one of
        }
    }
}

trait Trait<'a> {}

core::test_binder_constraints! {
    impl<'a> {
        dyn for<'b> Trait<'b>: 'a,
        //~^ ERROR the lhs of a ty outlives must be a placeholder
    }
}

core::test_binder_constraints! {
    impl<'a, T> {
        T: for<'b> Trait<'b>,
        //~^ ERROR expected lifetime, found keyword `for`
    }
}

core::test_binder_constraints! {
    impl<'a> {
        forall<T> where T: 'a {
            //~^ ERROR only lifetime parameters can be used in this context
            T: 'a,
            //~^ ERROR the lhs of a ty outlives must be a placeholder
        }
    }
}

fn main() {}
