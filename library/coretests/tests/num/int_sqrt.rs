macro_rules! tests {
    ($isqrt_consistency_check_fn_macro:ident : $($T:ident)+) => {
        $(
            mod $T {
                $isqrt_consistency_check_fn_macro!($T);

                // Check that the following produce the correct values from
                // `isqrt`:
                //
                // * the first and last 128 nonnegative values
                // * powers of two, minus one
                // * powers of two
                //
                // For signed types, check that `checked_isqrt` and `isqrt`
                // either produce the same numeric value or respectively
                // produce `None` and a panic. Make sure to do a consistency
                // check for `<$T>::MIN` as well, as no nonnegative values
                // negate to it.
                //
                // For unsigned types check that `isqrt` produces the same
                // numeric value for `$T` and `NonZero<$T>`.
                #[test]
                fn isqrt() {
                    isqrt_consistency_check(<$T>::MIN);

                    for n in (0..=127)
                        .chain(<$T>::MAX - 127..=<$T>::MAX)
                        .chain((0..<$T>::MAX.count_ones()).map(|exponent| (1 << exponent) - 1))
                        .chain((0..<$T>::MAX.count_ones()).map(|exponent| 1 << exponent))
                    {
                        isqrt_consistency_check(n);

                        let isqrt_n = n.isqrt();
                        assert!(
                            isqrt_n
                                .checked_mul(isqrt_n)
                                .map(|isqrt_n_squared| isqrt_n_squared <= n)
                                .unwrap_or(false),
                            "`{n}.isqrt()` should be lower than {isqrt_n}."
                        );
                        assert!(
                            (isqrt_n + 1)
                                .checked_mul(isqrt_n + 1)
                                .map(|isqrt_n_plus_1_squared| n < isqrt_n_plus_1_squared)
                                .unwrap_or(true),
                            "`{n}.isqrt()` should be higher than {isqrt_n})."
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
                    // square again to get to the next perfect square, minus
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
                    let mut n: $T = 0;
                    for sqrt_n in 0..1_024.min((1_u128 << (<$T>::MAX.count_ones()/2)) - 1) as $T {
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
                    //
                    // `MAX`'s `isqrt` return value is verified in the `isqrt`
                    // test function above.
                    let maximum_sqrt = <$T>::MAX.isqrt();
                    let mut n = maximum_sqrt * maximum_sqrt;

                    for sqrt_n in (maximum_sqrt - 1_024.min((1_u128 << (<$T>::MAX.count_ones()/2)) - 1) as $T..maximum_sqrt).rev() {
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

macro_rules! signed_check {
    ($T:ident) => {
        /// This takes an input and, if it's nonnegative or
        #[doc = concat!("`", stringify!($T), "::MIN`,")]
        /// checks that `isqrt` and `checked_isqrt` produce equivalent results
        /// for that input and for the negative of that input.
        ///
        /// # Note
        ///
        /// This cannot check that negative inputs to `isqrt` cause panics if
        /// panics abort instead of unwind.
        fn isqrt_consistency_check(n: $T) {
            // `<$T>::MIN` will be negative, so ignore it in this nonnegative
            // section.
            if n >= 0 {
                assert_eq!(
                    Some(n.isqrt()),
                    n.checked_isqrt(),
                    "`{n}.checked_isqrt()` should match `Some({n}.isqrt())`.",
                );
            }

            // `wrapping_neg` so that `<$T>::MIN` will negate to itself rather
            // than panicking.
            let negative_n = n.wrapping_neg();

            // Zero negated will still be nonnegative, so ignore it in this
            // negative section.
            if negative_n < 0 {
                assert_eq!(
                    negative_n.checked_isqrt(),
                    None,
                    "`({negative_n}).checked_isqrt()` should be `None`, as {negative_n} is negative.",
                );

                // `catch_unwind` only works when panics unwind rather than abort.
                #[cfg(panic = "unwind")]
                {
                    std::panic::catch_unwind(core::panic::AssertUnwindSafe(|| (-n).isqrt())).expect_err(
                        &format!("`({negative_n}).isqrt()` should have panicked, as {negative_n} is negative.")
                    );
                }
            }
        }
    };
}

macro_rules! unsigned_check {
    ($T:ident) => {
        /// This takes an input and, if it's nonzero, checks that `isqrt`
        /// produces the same numeric value for both
        #[doc = concat!("`", stringify!($T), "` and ")]
        #[doc = concat!("`NonZero<", stringify!($T), ">`.")]
        fn isqrt_consistency_check(n: $T) {
            // Zero cannot be turned into a `NonZero` value, so ignore it in
            // this nonzero section.
            if n > 0 {
                assert_eq!(
                    n.isqrt(),
                    core::num::NonZero::<$T>::new(n)
                        .expect(
                            "Was not able to create a new `NonZero` value from a nonzero number."
                        )
                        .isqrt()
                        .get(),
                    "`{n}.isqrt` should match `NonZero`'s `{n}.isqrt().get()`.",
                );
            }
        }
    };
}

tests!(signed_check: i8 i16 i32 i64 i128);
tests!(unsigned_check: u8 u16 u32 u64 u128);
