//@ compile-flags: -Zassumptions-on-binders
#![feature(test_binder_constraints)]
#![expect(incomplete_features)]

trait Trait<'a> {}
struct Struct<'a>(&'a u32);

core::test_binder_constraints! {
    impl<'a, 'b> {
        'a: 'b
        //~^ ERROR higher-ranked lifetime bound could not be satisfied
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
            //~| ERROR the lhs of a forall where clause must be an alias, placeholder, or lifetime
            T: 'a,
            //~^ ERROR the lhs of a ty outlives must be a placeholder
        }
    }
}

core::test_binder_constraints! {
    impl<'b, 'c: 'b + 'static> {
        forall<'a> where 'b: 'a {
            'c: 'a
        } expect {
            or {
                'c: 'b,
                'b: 'c,
                //~^ ERROR forall expect clause failed
            }
        }
    }
}

core::test_binder_constraints! {
    impl<'a, T> {
        for<> T: 'a
        //~^ ERROR bound type test binder constraint must be alias (it's a AliasTyOutlivesViaEnv)
    }
}

core::test_binder_constraints! {
    impl<'a> {
        forall<'b> where Struct<'a>: 'b {
            //~^ ERROR the lhs of a forall where clause must be an alias, placeholder, or lifetime
        }
    }
}

fn main() {}
