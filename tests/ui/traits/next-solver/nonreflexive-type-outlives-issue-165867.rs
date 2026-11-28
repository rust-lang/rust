//@ edition: 2024
//@ compile-flags: -Znext-solver=globally
//@ check-fail

#![crate_type = "lib"]
#![allow(dead_code)]

// Non-reflexive components still require outlives checking.

// A different lifetime must not be discarded.
mod distinct_region {
    trait Check<'a> {}

    impl<'a, T: 'a> Check<'a> for T {}

    trait Family<'a> {
        type Item;
    }

    fn demand<F>()
    where
        F: for<'a> Family<'a>,
        for<'a> <F as Family<'a>>::Item: Check<'a>,
    {}

    struct Carrier<T>(T);
    struct Wrap<'a, T>(&'a u8, T);

    impl<'a, 's> Family<'a> for Carrier<&'s u8> {
        type Item = Wrap<'a, &'s u8>;
    }

    fn test<'s>(_: &'s u8) {
        demand::<Carrier<&'s u8>>(); //~ ERROR lifetime may not live long enough
    }
}

// An unconstrained type parameter still requires an outlives bound.
mod generic_parameter {
    trait Check<'a> {}

    impl<'a, T: 'a> Check<'a> for T {}

    trait Family<'a> {
        type Item;
    }

    fn demand<F>()
    where
        F: for<'a> Family<'a>,
        for<'a> <F as Family<'a>>::Item: Check<'a>,
    {}

    struct Carrier<T>(T);
    struct Wrap<'a, T>(&'a u8, T);

    impl<'a, T> Family<'a> for Carrier<T> {
        type Item = Wrap<'a, T>;
    }

    fn test<T>() {
        demand::<Carrier<T>>(); //~ ERROR `T` does not live long enough
    }
}

// An associated type is not a reflexive region component.
mod projection {
    trait Check<'a> {}

    impl<'a, T: 'a> Check<'a> for T {}

    trait Family<'a> {
        type Item;
    }

    fn demand<F>()
    where
        F: for<'a> Family<'a>,
        for<'a> <F as Family<'a>>::Item: Check<'a>,
    {}

    struct Carrier<T>(T);
    struct Wrap<'a, T>(&'a u8, T);

    trait Project {
        type Assoc;
    }

    impl<'a, T: Project> Family<'a> for Carrier<T> {
        type Item = Wrap<'a, T::Assoc>;
    }

    fn test<T: Project>() {
        demand::<Carrier<T>>(); //~ ERROR does not live long enough
    }
}
