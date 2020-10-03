/// Creates an unsigned division function that uses a combination of hardware division and
/// binary long division to divide integers larger than what hardware division by itself can do. This
/// function is intended for microarchitectures that have division hardware, but not fast enough
/// multiplication hardware for `impl_trifecta` to be faster.
#[doc(hidden)]
#[macro_export]
macro_rules! impl_delegate {
    (
        $fn:ident, // name of the unsigned division function
        $zero_div_fn:ident, // function called when division by zero is attempted
        $half_normalization_shift:ident, // function for finding the normalization shift of $uX
        $half_division:ident, // function for division of a $uX by a $uX
        $n_h:expr, // the number of bits in $iH or $uH
        $uH:ident, // unsigned integer with half the bit width of $uX
        $uX:ident, // unsigned integer with half the bit width of $uD.
        $uD:ident, // unsigned integer type for the inputs and outputs of `$fn`
        $iD:ident // signed integer type with the same bitwidth as `$uD`
    ) => {
        /// Computes the quotient and remainder of `duo` divided by `div` and returns them as a
        /// tuple.
        pub fn $fn(duo: $uD, div: $uD) -> ($uD, $uD) {
            // The two possibility algorithm, undersubtracting long division algorithm, or any kind
            // of reciprocal based algorithm will not be fastest, because they involve large
            // multiplications that we assume to not be fast enough relative to the divisions to
            // outweigh setup times.

            // the number of bits in a $uX
            let n = $n_h * 2;

            let duo_lo = duo as $uX;
            let duo_hi = (duo >> n) as $uX;
            let div_lo = div as $uX;
            let div_hi = (div >> n) as $uX;

            match (div_lo == 0, div_hi == 0, duo_hi == 0) {
                (true, true, _) => $zero_div_fn(),
                (_, false, true) => {
                    // `duo` < `div`
                    return (0, duo);
                }
                (false, true, true) => {
                    // delegate to smaller division
                    let tmp = $half_division(duo_lo, div_lo);
                    return (tmp.0 as $uD, tmp.1 as $uD);
                }
                (false, true, false) => {
                    if duo_hi < div_lo {
                        // `quo_hi` will always be 0. This performs a binary long division algorithm
                        // to zero `duo_hi` followed by a half division.

                        // We can calculate the normalization shift using only `$uX` size functions.
                        // If we calculated the normalization shift using
                        // `$half_normalization_shift(duo_hi, div_lo false)`, it would break the
                        // assumption the function has that the first argument is more than the
                        // second argument. If the arguments are switched, the assumption holds true
                        // since `duo_hi < div_lo`.
                        let norm_shift = $half_normalization_shift(div_lo, duo_hi, false);
                        let shl = if norm_shift == 0 {
                            // Consider what happens if the msbs of `duo_hi` and `div_lo` align with
                            // no shifting. The normalization shift will always return
                            // `norm_shift == 0` regardless of whether it is fully normalized,
                            // because `duo_hi < div_lo`. In that edge case, `n - norm_shift` would
                            // result in shift overflow down the line. For the edge case, because
                            // both `duo_hi < div_lo` and we are comparing all the significant bits
                            // of `duo_hi` and `div`, we can make `shl = n - 1`.
                            n - 1
                        } else {
                            // We also cannot just use `shl = n - norm_shift - 1` in the general
                            // case, because when we are not in the edge case comparing all the
                            // significant bits, then the full `duo < div` may not be true and thus
                            // breaks the division algorithm.
                            n - norm_shift
                        };

                        // The 3 variable restoring division algorithm (see binary_long.rs) is ideal
                        // for this task, since `pow` and `quo` can be `$uX` and the delegation
                        // check is simple.
                        let mut div: $uD = div << shl;
                        let mut pow_lo: $uX = 1 << shl;
                        let mut quo_lo: $uX = 0;
                        let mut duo = duo;
                        loop {
                            let sub = duo.wrapping_sub(div);
                            if 0 <= (sub as $iD) {
                                duo = sub;
                                quo_lo |= pow_lo;
                                let duo_hi = (duo >> n) as $uX;
                                if duo_hi == 0 {
                                    // Delegate to get the rest of the quotient. Note that the
                                    // `div_lo` here is the original unshifted `div`.
                                    let tmp = $half_division(duo as $uX, div_lo);
                                    return ((quo_lo | tmp.0) as $uD, tmp.1 as $uD);
                                }
                            }
                            div >>= 1;
                            pow_lo >>= 1;
                        }
                    } else if duo_hi == div_lo {
                        // `quo_hi == 1`. This branch is cheap and helps with edge cases.
                        let tmp = $half_division(duo as $uX, div as $uX);
                        return ((1 << n) | (tmp.0 as $uD), tmp.1 as $uD);
                    } else {
                        // `div_lo < duo_hi`
                        // `rem_hi == 0`
                        if (div_lo >> $n_h) == 0 {
                            // Short division of $uD by a $uH, using $uX by $uX division
                            let div_0 = div_lo as $uH as $uX;
                            let (quo_hi, rem_3) = $half_division(duo_hi, div_0);

                            let duo_mid = ((duo >> $n_h) as $uH as $uX) | (rem_3 << $n_h);
                            let (quo_1, rem_2) = $half_division(duo_mid, div_0);

                            let duo_lo = (duo as $uH as $uX) | (rem_2 << $n_h);
                            let (quo_0, rem_1) = $half_division(duo_lo, div_0);

                            return (
                                (quo_0 as $uD) | ((quo_1 as $uD) << $n_h) | ((quo_hi as $uD) << n),
                                rem_1 as $uD,
                            );
                        }

                        // This is basically a short division composed of a half division for the hi
                        // part, specialized 3 variable binary long division in the middle, and
                        // another half division for the lo part.
                        let duo_lo = duo as $uX;
                        let tmp = $half_division(duo_hi, div_lo);
                        let quo_hi = tmp.0;
                        let mut duo = (duo_lo as $uD) | ((tmp.1 as $uD) << n);
                        // This check is required to avoid breaking the long division below.
                        if duo < div {
                            return ((quo_hi as $uD) << n, duo);
                        }

                        // The half division handled all shift alignments down to `n`, so this
                        // division can continue with a shift of `n - 1`.
                        let mut div: $uD = div << (n - 1);
                        let mut pow_lo: $uX = 1 << (n - 1);
                        let mut quo_lo: $uX = 0;
                        loop {
                            let sub = duo.wrapping_sub(div);
                            if 0 <= (sub as $iD) {
                                duo = sub;
                                quo_lo |= pow_lo;
                                let duo_hi = (duo >> n) as $uX;
                                if duo_hi == 0 {
                                    // Delegate to get the rest of the quotient. Note that the
                                    // `div_lo` here is the original unshifted `div`.
                                    let tmp = $half_division(duo as $uX, div_lo);
                                    return (
                                        (tmp.0) as $uD | (quo_lo as $uD) | ((quo_hi as $uD) << n),
                                        tmp.1 as $uD,
                                    );
                                }
                            }
                            div >>= 1;
                            pow_lo >>= 1;
                        }
                    }
                }
                (_, false, false) => {
                    // Full $uD by $uD binary long division. `quo_hi` will always be 0.
                    if duo < div {
                        return (0, duo);
                    }
                    let div_original = div;
                    let shl = $half_normalization_shift(duo_hi, div_hi, false);
                    let mut duo = duo;
                    let mut div: $uD = div << shl;
                    let mut pow_lo: $uX = 1 << shl;
                    let mut quo_lo: $uX = 0;
                    loop {
                        let sub = duo.wrapping_sub(div);
                        if 0 <= (sub as $iD) {
                            duo = sub;
                            quo_lo |= pow_lo;
                            if duo < div_original {
                                return (quo_lo as $uD, duo);
                            }
                        }
                        div >>= 1;
                        pow_lo >>= 1;
                    }
                }
            }
        }
    };
}
