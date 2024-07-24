//! This tests the `Integer::isqrt` methods.

macro_rules! tests {
    ($($SignedT:ident $UnsignedT:ident),+) => {
        $(
            mod $SignedT {
                /// This takes an input and, if it's nonnegative or
                #[doc = concat!("`", stringify!($SignedT), "::MIN`,")]
                /// checks that `isqrt` and `checked_isqrt` produce equivalent
                /// results for that input and for the negative of that input.
                fn isqrt_consistency_check(n: $SignedT) {
                    // `$SignedT::MIN` will be negative, so we don't want to handle `n` as if it's nonnegative.
                    if n >= 0 {
                        assert_eq!(
                            Some(n.isqrt()),
                            n.checked_isqrt(),
                            "`{n}.checked_isqrt()` should match `Some({n}.isqrt())`.",
                        );
                    }

                    // `wrapping_neg` so that `$SignedT::MIN` will negate to
                    // itself rather than panicking.
                    let negative_n = n.wrapping_neg();

                    // `negative_n` should be negative, but `n` could be zero,
                    // so make sure not to check that one.
                    if negative_n < 0 {
                        assert_eq!(
                            negative_n.checked_isqrt(),
                            None,
                            "`({negative_n}).checked_isqrt()` should be `None`, as {negative_n} is negative.",
                        );

                        ::std::panic::catch_unwind(::core::panic::AssertUnwindSafe(|| (-n).isqrt())).expect_err(
                            &format!("`({negative_n}).isqrt()` should have panicked, as {negative_n} is negative.")
                        );
                    }
                }

                // Check that the following produce the correct values from
                // `isqrt` and that `checked_isqrt` produces the same numeric
                // value as `isqrt`. Check also that their negative versions
                // and `$SignedT::MIN` produce a panic from `isqrt` and `None`
                // from `checked_isqrt`:
                //
                // * the first and last 128 nonnegative values
                // * powers of two, minus one
                // * powers of two
                #[test]
                fn isqrt() {
                    // Check the minimum value because there's no positive
                    // value that can be negated into the minimum value.
                    isqrt_consistency_check($SignedT::MIN);

                    for n in (0..=127)
                        .chain($SignedT::MAX - 127..=$SignedT::MAX)
                        .chain((0..$SignedT::BITS - 1).map(|exponent| (1 << exponent) - 1))
                        .chain((0..$SignedT::BITS - 1).map(|exponent| 1 << exponent))
                    {
                        isqrt_consistency_check(n);

                        let sqrt_n = n.isqrt();
                        assert!(
                            sqrt_n * sqrt_n <= n,
                            "The integer square root of {n} should be lower than {sqrt_n} (the current return value of `{n}.isqrt()`)."
                        );
                        assert!(
                            (sqrt_n + 1).checked_mul(sqrt_n + 1).map(|higher_than_n| n < higher_than_n).unwrap_or(true),
                            "The integer square root of {n} should be higher than {sqrt_n} (the current return value of `{n}.isqrt()`)."
                        );
                    }
                }

                // Check the square roots of:
                //
                // * the first 1,024 perfect squares
                // * halfway between each of the first 1,024 perfect squares
                //   and the next perfect square
                // * the next perfect square after the each of the first 1,024
                //   perfect squares, minus one
                // * the last 1,024 perfect squares
                // * the last 1,024 perfect squares, minus one
                // * halfway between each of the last 1,024 perfect squares
                //   and the previous perfect square
                #[test]
                // Skip this test on Miri, as it takes too long to run.
                #[cfg(not(miri))]
                fn isqrt_extended() {
                    // The correct value is worked out by using the fact that
                    // the nth nonzero perfect square is the sum of the first n
                    // odd numbers:
                    //
                    //  1 = 1
                    //  4 = 1 + 3
                    //  9 = 1 + 3 + 5
                    // 16 = 1 + 3 + 5 + 7
                    //
                    // Note also that the last odd number added in is two times
                    // the square root of the previous perfect square, plus
                    // one:
                    //
                    // 1 = 2*0 + 1
                    // 3 = 2*1 + 1
                    // 5 = 2*2 + 1
                    // 7 = 2*3 + 1
                    //
                    // That means we can add the square root of this perfect
                    // square once to get about halfway to the next perfect
                    // square, then we can add the square root of this perfect
                    // square again to get to the next perfect square minus
                    // one, then we can add one to get to the next perfect
                    // square.
                    //
                    // This allows us to, for each of the first 1,024 perfect
                    // squares, test that the square roots of the following are
                    // all correct and equal to each other:
                    //
                    // * the current perfect square
                    // * about halfway to the next perfect square
                    // * the next perfect square, minus one
                    let mut n: $SignedT = 0;
                    for sqrt_n in 0..1_024.min((1_u128 << (($SignedT::BITS - 1)/2)) - 1) as $SignedT {
                        isqrt_consistency_check(n);
                        assert_eq!(
                            n.isqrt(),
                            sqrt_n,
                            "`{sqrt_n}.pow(2).isqrt()` should be {sqrt_n}."
                        );

                        n += sqrt_n;
                        isqrt_consistency_check(n);
                        assert_eq!(
                            n.isqrt(),
                            sqrt_n,
                            "{n} is about halfway between `{sqrt_n}.pow(2)` and `{}.pow(2)`, so `{n}.isqrt()` should be {sqrt_n}.",
                            sqrt_n + 1
                        );

                        n += sqrt_n;
                        isqrt_consistency_check(n);
                        assert_eq!(
                            n.isqrt(),
                            sqrt_n,
                            "`({}.pow(2) - 1).isqrt()` should be {sqrt_n}.",
                            sqrt_n + 1
                        );

                        n += 1;
                    }

                    // Similarly, for each of the last 1,024 perfect squares,
                    // check:
                    //
                    // * the current perfect square
                    // * the current perfect square, minus one
                    // * about halfway to the previous perfect square

                    // `MAX`'s `isqrt` return value verified in `isqrt` test
                    // function above.
                    let maximum_sqrt = $SignedT::MAX.isqrt();
                    let mut n = maximum_sqrt * maximum_sqrt;

                    for sqrt_n in (maximum_sqrt - 1_024.min((1_u128 << (($SignedT::BITS - 1)/2)) - 1) as $SignedT..maximum_sqrt).rev() {
                        isqrt_consistency_check(n);
                        assert_eq!(
                            n.isqrt(),
                            sqrt_n + 1,
                            "`{0}.pow(2).isqrt()` should be {0}.",
                            sqrt_n + 1
                        );

                        n -= 1;
                        isqrt_consistency_check(n);
                        assert_eq!(
                            n.isqrt(),
                            sqrt_n,
                            "`({}.pow(2) - 1).isqrt()` should be {sqrt_n}.",
                            sqrt_n + 1
                        );

                        n -= sqrt_n;
                        isqrt_consistency_check(n);
                        assert_eq!(
                            n.isqrt(),
                            sqrt_n,
                            "{n} is about halfway between `{sqrt_n}.pow(2)` and `{}.pow(2)`, so `{n}.isqrt()` should be {sqrt_n}.",
                            sqrt_n + 1
                        );

                        n -= sqrt_n;
                    }
                }
            }

            mod $UnsignedT {
                /// This takes an input and, if it's nonzero, checks that
                /// `isqrt` produces the same numeric value for both
                #[doc = concat!("`", stringify!($UnsignedT), "` and ")]
                #[doc = concat!("`NonZero<", stringify!($UnsignedT), ">`.")]
                fn isqrt_consistency_check(n: $UnsignedT) {
                    if n > 0 {
                        assert_eq!(
                            n.isqrt(),
                            ::core::num::NonZero::<$UnsignedT>::new(n)
                                .expect("Cannot create a new `NonZero` value from a nonzero value")
                                .isqrt()
                                .get(),
                            "`{n}.isqrt` should match `NonZero`'s `{n}.isqrt().get()`.",
                        );
                    }
                }

                // Check that the following produce the correct values from
                // `isqrt` and that `checked_isqrt` produces the same numeric
                // value as `isqrt`:
                //
                // * the first and last 128 values
                // * powers of two, minus one
                // * powers of two
                #[test]
                fn isqrt() {
                    for n in (0..=127)
                        .chain($UnsignedT::MAX - 127..=$UnsignedT::MAX)
                        .chain((0..$UnsignedT::BITS).map(|exponent| (1 << exponent) - 1))
                        .chain((0..$UnsignedT::BITS).map(|exponent| 1 << exponent))
                    {
                        isqrt_consistency_check(n);

                        let sqrt_n = n.isqrt();
                        assert!(
                            sqrt_n * sqrt_n <= n,
                            "The integer square root of {n} should be lower than {sqrt_n} (the current return value of `{n}.isqrt()`)."
                        );
                        assert!(
                            (sqrt_n + 1).checked_mul(sqrt_n + 1).map(|higher_than_n| n < higher_than_n).unwrap_or(true),
                            "The integer square root of {n} should be higher than {sqrt_n} (the current return value of `{n}.isqrt()`)."
                        );
                    }
                }

                // Check the square roots of:
                //
                // * the first 1,024 perfect squares
                // * halfway between each of the first 1,024 perfect squares
                //   and the next perfect square
                // * the next perfect square after the each of the first 1,024
                //   perfect squares, minus one
                // * the last 1,024 perfect squares
                // * the last 1,024 perfect squares, minus one
                // * halfway between each of the last 1,024 perfect squares
                //   and the previous perfect square
                #[test]
                // Skip this test on Miri, as it takes too long to run.
                #[cfg(not(miri))]
                fn test_isqrt_extended() {
                    // The correct value is worked out by using the fact that
                    // the nth nonzero perfect square is the sum of the first n
                    // odd numbers:
                    //
                    //  1 = 1
                    //  4 = 1 + 3
                    //  9 = 1 + 3 + 5
                    // 16 = 1 + 3 + 5 + 7
                    //
                    // Note also that the last odd number added in is two times
                    // the square root of the previous perfect square, plus
                    // one:
                    //
                    // 1 = 2*0 + 1
                    // 3 = 2*1 + 1
                    // 5 = 2*2 + 1
                    // 7 = 2*3 + 1
                    //
                    // That means we can add the square root of this perfect
                    // square once to get about halfway to the next perfect
                    // square, then we can add the square root of this perfect
                    // square again to get to the next perfect square minus
                    // one, then we can add one to get to the next perfect
                    // square.
                    //
                    // This allows us to, for each of the first 1,024 perfect
                    // squares, test that the square roots of the following are
                    // all correct and equal to each other:
                    //
                    // * the current perfect square
                    // * about halfway to the next perfect square
                    // * the next perfect square, minus one
                    let mut n: $UnsignedT = 0;
                    for sqrt_n in 0..1_024.min((1_u128 << ($UnsignedT::BITS/2)) - 1) as $UnsignedT {
                        isqrt_consistency_check(n);
                        assert_eq!(
                            n.isqrt(),
                            sqrt_n,
                            "`{sqrt_n}.pow(2).isqrt()` should be {sqrt_n}."
                        );

                        n += sqrt_n;
                        isqrt_consistency_check(n);
                        assert_eq!(
                            n.isqrt(),
                            sqrt_n,
                            "{n} is about halfway between `{sqrt_n}.pow(2)` and `{}.pow(2)`, so `{n}.isqrt()` should be {sqrt_n}.",
                            sqrt_n + 1
                        );

                        n += sqrt_n;
                        isqrt_consistency_check(n);
                        assert_eq!(
                            n.isqrt(),
                            sqrt_n,
                            "`({}.pow(2) - 1).isqrt()` should be {sqrt_n}.",
                            sqrt_n + 1
                        );

                        n += 1;
                    }

                    // Similarly, for each of the last 1,024 perfect squares,
                    // check:
                    //
                    // * the current perfect square
                    // * the current perfect square, minus one
                    // * about halfway to the previous perfect square

                    // `MAX`'s `isqrt` return value verified in `isqrt` test
                    // function above.
                    let maximum_sqrt = $UnsignedT::MAX.isqrt();
                    let mut n = maximum_sqrt * maximum_sqrt;

                    for sqrt_n in (maximum_sqrt - 1_024.min((1_u128 << ($UnsignedT::BITS/2)) - 1) as $UnsignedT..maximum_sqrt).rev() {
                        isqrt_consistency_check(n);
                        assert_eq!(
                            n.isqrt(),
                            sqrt_n + 1,
                            "`{0}.pow(2).isqrt()` should be {0}.",
                            sqrt_n + 1
                        );

                        n -= 1;
                        isqrt_consistency_check(n);
                        assert_eq!(
                            n.isqrt(),
                            sqrt_n,
                            "`({}.pow(2) - 1).isqrt()` should be {sqrt_n}.",
                            sqrt_n + 1
                        );

                        n -= sqrt_n;
                        isqrt_consistency_check(n);
                        assert_eq!(
                            n.isqrt(),
                            sqrt_n,
                            "{n} is about halfway between `{sqrt_n}.pow(2)` and `{}.pow(2)`, so `{n}.isqrt()` should be {sqrt_n}.",
                            sqrt_n + 1
                        );

                        n -= sqrt_n;
                    }
                }
            }
        )*
    };
}

tests!(i8 u8, i16 u16, i32 u32, i64 u64, i128 u128, isize usize);
