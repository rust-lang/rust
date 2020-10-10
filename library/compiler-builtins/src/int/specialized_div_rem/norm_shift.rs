/// Creates a function used by some division algorithms to compute the "normalization shift".
#[doc(hidden)]
#[macro_export]
macro_rules! impl_normalization_shift {
    (
        $name:ident, // name of the normalization shift function
        // boolean for if `$uX::leading_zeros` should be used (if an architecture does not have a
        // hardware instruction for `usize::leading_zeros`, then this should be `true`)
        $use_lz:ident,
        $n:tt, // the number of bits in a $iX or $uX
        $uX:ident, // unsigned integer type for the inputs of `$name`
        $iX:ident, // signed integer type for the inputs of `$name`
        $($unsigned_attr:meta),* // attributes for the function
    ) => {
        /// Finds the shift left that the divisor `div` would need to be normalized for a binary
        /// long division step with the dividend `duo`. NOTE: This function assumes that these edge
        /// cases have been handled before reaching it:
        /// `
        /// if div == 0 {
        ///     panic!("attempt to divide by zero")
        /// }
        /// if duo < div {
        ///     return (0, duo)
        /// }
        /// `
        ///
        /// Normalization is defined as (where `shl` is the output of this function):
        /// `
        /// if duo.leading_zeros() != (div << shl).leading_zeros() {
        ///     // If the most significant bits of `duo` and `div << shl` are not in the same place,
        ///     // then `div << shl` has one more leading zero than `duo`.
        ///     assert_eq!(duo.leading_zeros() + 1, (div << shl).leading_zeros());
        ///     // Also, `2*(div << shl)` is not more than `duo` (otherwise the first division step
        ///     // would not be able to clear the msb of `duo`)
        ///     assert!(duo < (div << (shl + 1)));
        /// }
        /// if full_normalization {
        ///     // Some algorithms do not need "full" normalization, which means that `duo` is
        ///     // larger than `div << shl` when the most significant bits are aligned.
        ///     assert!((div << shl) <= duo);
        /// }
        /// `
        ///
        /// Note: If the software bisection algorithm is being used in this function, it happens
        /// that full normalization always occurs, so be careful that new algorithms are not
        /// invisibly depending on this invariant when `full_normalization` is set to `false`.
        $(
            #[$unsigned_attr]
        )*
        fn $name(duo: $uX, div: $uX, full_normalization: bool) -> usize {
            // We have to find the leading zeros of `div` to know where its msb (most significant
            // set bit) is to even begin binary long division. It is also good to know where the msb
            // of `duo` is so that useful work can be started instead of shifting `div` for all
            // possible quotients (many division steps are wasted if `duo.leading_zeros()` is large
            // and `div` starts out being shifted all the way to the msb). Aligning the msbs of
            // `div` and `duo` could be done by shifting `div` left by
            // `div.leading_zeros() - duo.leading_zeros()`, but some CPUs without division hardware
            // also do not have single instructions for calculating `leading_zeros`. Instead of
            // software doing two bisections to find the two `leading_zeros`, we do one bisection to
            // find `div.leading_zeros() - duo.leading_zeros()` without actually knowing either of
            // the leading zeros values.

            let mut shl: usize;
            if $use_lz {
                shl = (div.leading_zeros() - duo.leading_zeros()) as usize;
                if full_normalization {
                    if duo < (div << shl) {
                        // when the msb of `duo` and `div` are aligned, the resulting `div` may be
                        // larger than `duo`, so we decrease the shift by 1.
                        shl -= 1;
                    }
                }
            } else {
                let mut test = duo;
                shl = 0usize;
                let mut lvl = $n >> 1;
                loop {
                    let tmp = test >> lvl;
                    // It happens that a final `duo < (div << shl)` check is not needed, because the
                    // `div <= tmp` check insures that the msb of `test` never passes the msb of
                    // `div`, and any set bits shifted off the end of `test` would still keep
                    // `div <= tmp` true.
                    if div <= tmp {
                        test = tmp;
                        shl += lvl;
                    }
                    // narrow down bisection
                    lvl >>= 1;
                    if lvl == 0 {
                        break
                    }
                }
            }
            // tests the invariants that should hold before beginning binary long division
            /*
            if full_normalization {
                assert!((div << shl) <= duo);
            }
            if duo.leading_zeros() != (div << shl).leading_zeros() {
                assert_eq!(duo.leading_zeros() + 1, (div << shl).leading_zeros());
                assert!(duo < (div << (shl + 1)));
            }
            */
            shl
        }
    }
}
