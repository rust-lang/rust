//@ edition: 2024
//@ compile-flags: -Znext-solver=globally
//@ check-pass

#![crate_type = "lib"]
#![allow(dead_code)]

// Reflexive type-outlives constraints are not specific to coroutines
// or placeholder regions.

// An ADT containing only the required placeholder region.
mod placeholder {
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

    impl<'a> Family<'a> for () {
        type Item = Wrap<'a, &'a u32>;
    }

    fn test() {
        demand::<()>();
    }
}

// The same relation with an early-bound lifetime.
mod early_bound {
    trait Check<'a> {}

    impl<'a, T: 'a> Check<'a> for T {}

    trait Family<'a> {
        type Item;
    }

    fn demand<'a, F>()
    where
        F: Family<'a>,
        F::Item: Check<'a>,
    {}

    impl<'a> Family<'a> for () {
        type Item = (&'a u8, &'a u32);
    }

    fn test<'a>(_: &'a u8)
    where
        'a: 'a,
    {
        demand::<'a, ()>();
    }
}

// A locally bound lifetime does not escape its binder.
mod nested_binder {
    trait Check<'a> {}

    impl<'a, T: 'a> Check<'a> for T {}

    trait Family<'a> {
        type Item;
    }

    fn demand<'a, F>()
    where
        F: Family<'a>,
        F::Item: Check<'a>,
    {}

    impl<'a> Family<'a> for () {
        type Item = (&'a u8, for<'b> fn(&'b u8));
    }

    fn test<'a>(_: &'a u8) {
        demand::<'a, ()>();
    }
}
