//@ ignore-compare-mode-polonius (explicit revisions)
//@ edition: 2024
//@ revisions: nll polonius
//@[nll] compile-flags: -Z polonius=off
//@[polonius] compile-flags: -Z polonius=next

// When we require that an alias outlives some lifetime `'x`, we may register a
// type test. Once we *prove* that type test, we ned to register that proof at
// the correct location for Polonius Alpha.

#![forbid(unsafe_code)]

fn require_static<T: 'static>(_: T) {}

mod simple {
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

mod source_not_a_bound {
    use super::require_static;
    use std::fmt::Debug;

    fn make<'a, 'b, 'c, 'z>(x: &'c u8, _z: &'z u8) -> impl Debug + Copy + 'a + 'b
    where
        'c: 'a,
        'c: 'b,
        'z: 'z,
    {
        x
    }

    pub fn test() {
        let z = 0u8;
        let mut y = make::<'static, 'static, 'static, '_>(&0, &z);
        {
            let x = Box::new(42u8);
            y = make(&*x, &z);
            //~^ ERROR `*x` does not live long enough
            require_static(y);
        }
    }
}

mod two_sources {
    use super::require_static;

    fn make<'a, 'b, 'c, 'd>(c: &'c u8, d: &'d u8) -> impl Copy + 'a + 'b
    where
        'c: 'a,
        'c: 'b,
        'd: 'a,
        'd: 'b,
    {
        let _ = c;
        d
    }

    pub fn test() {
        let mut value = make::<'static, 'static, 'static, 'static>(&0, &0);
        {
            let d = 42u8;
            value = make(&0, &d);
            //~^ ERROR `d` does not live long enough
            require_static(value);
        }
    }
}

mod projection {
    use super::require_static;

    pub trait Tr<'a: 'b, 'b> {
        type A: Copy + 'a + 'b;
        fn mk(x: &'a u8) -> Self::A;
    }

    impl<'a> Tr<'a, 'a> for () {
        type A = &'a u8;

        fn mk(x: &'a u8) -> Self::A {
            x
        }
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

mod no_type_test {
    fn make<'a: 'b, 'b>(x: &'a String) -> impl Copy + 'a + 'b {
        x
    }

    pub fn test() {
        let x = String::new();
        let y = String::new();
        let mut value = make(&x);
        value = make(&y);
        drop(x);
        let _ = value;
    }
}

mod env_clause {
    pub trait Tr {
        type Assoc<'x>;
        fn make<'x>(x: &'x u8) -> Self::Assoc<'x>;
    }

    impl Tr for () {
        type Assoc<'x> = ();

        fn make<'x>(_: &'x u8) -> Self::Assoc<'x> {}
    }

    fn require<'p, T: 'p>(_: T) {}

    pub fn test<'p, T: Tr>()
    where
        for<'x> T::Assoc<'x>: 'p,
    {
        let x = 0u8;
        require::<'p, _>(T::make(&x));
    }
}


fn main() {
    simple::test();
    source_not_a_bound::test();
    two_sources::test();
    // Commented out because this needs higher-ranked assumptions
    //projection::test::<()>();
    no_type_test::test();
    env_clause::test::<()>();
}
