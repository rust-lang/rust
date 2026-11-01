// Proving that an alias outlives a lifetime is not always straightforward. If
// there is a single outlives bound, then we can simply register an outlives
// constraint. However, if there are multiple bounds, then *any* of them ca be
// used to prove the outlives relationship; so, we register a type test instead.
//
// Under NLL, this is not a problem. The bounds used to prove the type test will
// have been implied through some other point in the program; and because NLL
// is location-insensitive, the outlives relationship will be implied at the
// point of the type test.
//
// Under Polonius Alpha, the outlives relationship is location-sensitive, it is
// not enough for the outlives relationship to be implied at some other point
// in the program. Rather, we need to register this constraint *at the point of
// the type test*.

//@ ignore-compare-mode-polonius (explicit revisions)
//@ edition: 2024
//@ revisions: nll polonius
//@ [nll] compile-flags: -Z polonius=off
//@ [polonius] compile-flags: -Z polonius=next
#![forbid(unsafe_code)]

fn require_static<T: 'static>(_: T) {}

// If there are two outlives bounds on an alias, then we can't simply register
// an outlives constraint. Instead, a `AnyBound` type test is registered. This
// test shows observable UB.
mod observable_ub {
    use std::fmt::Debug;

    fn make<'a: 'b, 'b>(x: &'a u8) -> impl Debug + Copy + 'a + 'b {
        x
    }

    pub fn test() {
        let mut y = make::<'static, 'static>(&0);
        let escaped: Box<dyn Debug>;
        {
            let x = 42u8;
            y = make(&x);
            //~^ ERROR `x` does not live long enough
            escaped = Box::new(y);
        }
        println!("{escaped:?}");
    }
}

// A more minimal form of the above, but requires the alias to outlive `'static`
// rather than being used outside the inner block.
mod ordered_bounds {
    use super::require_static;

    fn make<'a: 'b, 'b>(x: &'a u8) -> impl Copy + 'a + 'b {
        x
    }

    pub fn test() {
        let mut y = make::<'static, 'static>(&0);
        {
            let x = 42u8;
            y = make(&x);
            //~^ ERROR `x` does not live long enough
            require_static(y);
        }
    }
}

// Unlike the previous two (which has a direct outlives relationship between
// `'a` and `'b`), this test doesn't and so a type test *must* be used.
mod unrelated_bounds {
    use super::require_static;

    fn make<'a, 'b, 'c>(x: &'c u8) -> impl Copy + 'a + 'b
    where
        'c: 'a,
        'c: 'b,
    {
        x
    }

    pub fn test() {
        let mut y = make::<'static, 'static, 'static>(&0);
        {
            let x = 42u8;
            y = make(&x);
            //~^ ERROR `x` does not live long enough
            require_static(y);
        }
    }
}

// Any universal region has the same effect as `'static`.
mod universal_param {
    fn make<'a: 'b, 'b>(x: &'a u8) -> impl Copy + 'a + 'b {
        x
    }

    fn require<'p, T: 'p>(_: T) {}

    pub fn test<'p>(seed: &'p u8) {
        let mut y = make::<'p, 'p>(seed);
        {
            let x = 42u8;
            y = make(&x);
            //~^ ERROR `x` does not live long enough
            require::<'p, _>(y);
        }
    }
}

// The same behavior applies to projections, too.
mod projection {
    use super::require_static;

    pub trait Tr<'a: 'b, 'b> {
        type A: Copy + 'a + 'b;
        fn mk(x: &'a u8) -> Self::A;
    }

    pub fn test<T: for<'a, 'b> Tr<'a, 'b>>() {
        let mut y = <T as Tr<'static, 'static>>::mk(&0);
        {
            let x = 42u8;
            y = <T as Tr<'_, '_>>::mk(&x);
            //~^ ERROR `x` does not live long enough
            require_static(y);
        }
    }
}


// Like `ordered_bounds`, but doesn't actually *require* that the alias outlives
// `'static`. NLL fails here, but polonius can pass because of location-sensitive
// outlives bounds.
mod no_bounds {
    fn make<'a: 'b, 'b>(x: &'a u8) -> impl Copy + 'a + 'b {
        x
    }

    pub fn test() {
        let mut y = make::<'static, 'static>(&0);
        {
            let x = 42u8;
            y = make(&x);
            //[nll]~^ ERROR `x` does not live long enough
        }
    }
}

fn main() {
    observable_ub::test();
}
