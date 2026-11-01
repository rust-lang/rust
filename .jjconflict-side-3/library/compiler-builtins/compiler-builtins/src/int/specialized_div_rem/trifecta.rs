/// Creates an unsigned division function optimized for division of integers with bitwidths
/// larger than the largest hardware integer division supported. These functions use large radix
/// division algorithms that require both fast division and very fast widening multiplication on the
/// target microarchitecture. Otherwise, `impl_delegate` should be used instead.
#[allow(unused_macros)]
macro_rules! impl_trifecta {
    (
        $fn:ident, // name of the unsigned division function
        $zero_div_fn:ident, // function called when division by zero is attempted
        $half_division:ident, // function for division of a $uX by a $uX
        $n_h:expr, // the number of bits in $iH or $uH
        $uH:ident, // unsigned integer with half the bit width of $uX
        $uX:ident, // unsigned integer with half the bit width of $uD
        $uD:ident // unsigned integer type for the inputs and outputs of `$unsigned_name`
    ) => {
        /// Computes the quotient and remainder of `duo` divided by `div` and returns them as a
        /// tuple.
        pub fn $fn(duo: $uD, div: $uD) -> ($uD, $uD) {
            // This is called the trifecta algorithm because it uses three main algorithms: short
            // division for small divisors, the two possibility algorithm for large divisors, and an
            // undersubtracting long division algorithm for intermediate cases.

            // This replicates `carrying_mul` (rust-lang rfc #2417). LLVM correctly optimizes this
            // to use a widening multiply to 128 bits on the relevant architectures.
            fn carrying_mul(lhs: $uX, rhs: $uX) -> ($uX, $uX) {
                let tmp = (lhs as $uD).wrapping_mul(rhs as $uD);
                (tmp as $uX, (tmp >> ($n_h * 2)) as $uX)
            }
            fn carrying_mul_add(lhs: $uX, mul: $uX, add: $uX) -> ($uX, $uX) {
                let tmp = (lhs as $uD)
                    .wrapping_mul(mul as $uD)
                    .wrapping_add(add as $uD);
                (tmp as $uX, (tmp >> ($n_h * 2)) as $uX)
            }

            // the number of bits in a $uX
            let n = $n_h * 2;

            if div == 0 {
                $zero_div_fn()
            }

            // Trying to use a normalization shift function will cause inelegancies in the code and
            // inefficiencies for architectures with a native count leading zeros instruction. The
            // undersubtracting algorithm needs both values (keeping the original `div_lz` but
            // updating `duo_lz` multiple times), so we assume hardware support for fast
            // `leading_zeros` calculation.
            let div_lz = div.leading_zeros();
            let mut duo_lz = duo.leading_zeros();

            // the possible ranges of `duo` and `div` at this point:
            // `0 <= duo < 2^n_d`
            // `1 <= div < 2^n_d`

            // quotient is 0 or 1 branch
            if div_lz <= duo_lz {
                // The quotient cannot be more than 1. The highest set bit of `duo` needs to be at
                // least one place higher than `div` for the quotient to be more than 1.
                if duo >= div {
                    return (1, duo - div);
                } else {
                    return (0, duo);
                }
            }

            // `_sb` is the number of significant bits (from the ones place to the highest set bit)
            // `{2, 2^div_sb} <= duo < 2^n_d`
            // `1 <= div < {2^duo_sb, 2^(n_d - 1)}`
            // smaller division branch
            if duo_lz >= n {
                // `duo < 2^n` so it will fit in a $uX. `div` will also fit in a $uX (because of the
                // `div_lz <= duo_lz` branch) so no numerical error.
                let (quo, rem) = $half_division(duo as $uX, div as $uX);
                return (quo as $uD, rem as $uD);
            }

            // `{2^n, 2^div_sb} <= duo < 2^n_d`
            // `1 <= div < {2^duo_sb, 2^(n_d - 1)}`
            // short division branch
            if div_lz >= (n + $n_h) {
                // `1 <= div < {2^duo_sb, 2^n_h}`

                // It is barely possible to improve the performance of this by calculating the
                // reciprocal and removing one `$half_division`, but only if the CPU can do fast
                // multiplications in parallel. Other reciprocal based methods can remove two
                // `$half_division`s, but have multiplications that cannot be done in parallel and
                // reduce performance. I have decided to use this trivial short division method and
                // rely on the CPU having quick divisions.

                let duo_hi = (duo >> n) as $uX;
                let div_0 = div as $uH as $uX;
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

            // relative leading significant bits, cannot overflow because of above branches
            let lz_diff = div_lz - duo_lz;

            // `{2^n, 2^div_sb} <= duo < 2^n_d`
            // `2^n_h <= div < {2^duo_sb, 2^(n_d - 1)}`
            // `mul` or `mul - 1` branch
            if lz_diff < $n_h {
                // Two possibility division algorithm

                // The most significant bits of `duo` and `div` are within `$n_h` bits of each
                // other. If we take the `n` most significant bits of `duo` and divide them by the
                // corresponding bits in `div`, it produces a quotient value `quo`. It happens that
                // `quo` or `quo - 1` will always be the correct quotient for the whole number. In
                // other words, the bits less significant than the `n` most significant bits of
                // `duo` and `div` can only influence the quotient to be one of two values.
                // Because there are only two possibilities, there only needs to be one `$uH` sized
                // division, a `$uH` by `$uD` multiplication, and only one branch with a few simple
                // operations.
                //
                // Proof that the true quotient can only be `quo` or `quo - 1`.
                // All `/` operators here are floored divisions.
                //
                // `shift` is the number of bits not in the higher `n` significant bits of `duo`.
                // (definitions)
                // 0. shift = n - duo_lz
                // 1. duo_sig_n == duo / 2^shift
                // 2. div_sig_n == div / 2^shift
                // 3. quo == duo_sig_n / div_sig_n
                //
                //
                // We are trying to find the true quotient, `true_quo`.
                // 4. true_quo = duo / div. (definition)
                //
                // This is true because of the bits that are cut off during the bit shift.
                // 5. duo_sig_n * 2^shift <= duo < (duo_sig_n + 1) * 2^shift.
                // 6. div_sig_n * 2^shift <= div < (div_sig_n + 1) * 2^shift.
                //
                // Dividing each bound of (5) by each bound of (6) gives 4 possibilities for what
                // `true_quo == duo / div` is bounded by:
                // (duo_sig_n * 2^shift) / (div_sig_n * 2^shift)
                // (duo_sig_n * 2^shift) / ((div_sig_n + 1) * 2^shift)
                // ((duo_sig_n + 1) * 2^shift) / (div_sig_n * 2^shift)
                // ((duo_sig_n + 1) * 2^shift) / ((div_sig_n + 1) * 2^shift)
                //
                // Simplifying each of these four:
                // duo_sig_n / div_sig_n
                // duo_sig_n / (div_sig_n + 1)
                // (duo_sig_n + 1) / div_sig_n
                // (duo_sig_n + 1) / (div_sig_n + 1)
                //
                // Taking the smallest and the largest of these as the low and high bounds
                // and replacing `duo / div` with `true_quo`:
                // 7. duo_sig_n / (div_sig_n + 1) <= true_quo < (duo_sig_n + 1) / div_sig_n
                //
                // The `lz_diff < n_h` conditional on this branch makes sure that `div_sig_n` is at
                // least `2^n_h`, and the `div_lz <= duo_lz` branch makes sure that the highest bit
                // of `div_sig_n` is not the `2^(n - 1)` bit.
                // 8. `2^(n - 1) <= duo_sig_n < 2^n`
                // 9. `2^n_h <= div_sig_n < 2^(n - 1)`
                //
                // We want to prove that either
                // `(duo_sig_n + 1) / div_sig_n == duo_sig_n / (div_sig_n + 1)` or that
                // `(duo_sig_n + 1) / div_sig_n == duo_sig_n / (div_sig_n + 1) + 1`.
                //
                // We also want to prove that `quo` is one of these:
                // `duo_sig_n / div_sig_n == duo_sig_n / (div_sig_n + 1)` or
                // `duo_sig_n / div_sig_n == (duo_sig_n + 1) / div_sig_n`.
                //
                // When 1 is added to the numerator of `duo_sig_n / div_sig_n` to produce
                // `(duo_sig_n + 1) / div_sig_n`, it is not possible that the value increases by
                // more than 1 with floored integer arithmetic and `div_sig_n != 0`. Consider
                // `x/y + 1 < (x + 1)/y` <=> `x/y + 1 < x/y + 1/y` <=> `1 < 1/y` <=> `y < 1`.
                // `div_sig_n` is a nonzero integer. Thus,
                // 10. `duo_sig_n / div_sig_n == (duo_sig_n + 1) / div_sig_n` or
                //     `(duo_sig_n / div_sig_n) + 1 == (duo_sig_n + 1) / div_sig_n.
                //
                // When 1 is added to the denominator of `duo_sig_n / div_sig_n` to produce
                // `duo_sig_n / (div_sig_n + 1)`, it is not possible that the value decreases by
                // more than 1 with the bounds (8) and (9). Consider `x/y - 1 <= x/(y + 1)` <=>
                // `(x - y)/y < x/(y + 1)` <=> `(y + 1)*(x - y) < x*y` <=> `x*y - y*y + x - y < x*y`
                // <=> `x < y*y + y`. The smallest value of `div_sig_n` is `2^n_h` and the largest
                // value of `duo_sig_n` is `2^n - 1`. Substituting reveals `2^n - 1 < 2^n + 2^n_h`.
                // Thus,
                // 11. `duo_sig_n / div_sig_n == duo_sig_n / (div_sig_n + 1)` or
                //     `(duo_sig_n / div_sig_n) - 1` == duo_sig_n / (div_sig_n + 1)`
                //
                // Combining both (10) and (11), we know that
                // `quo - 1 <= duo_sig_n / (div_sig_n + 1) <= true_quo
                // < (duo_sig_n + 1) / div_sig_n <= quo + 1` and therefore:
                // 12. quo - 1 <= true_quo < quo + 1
                //
                // In a lot of division algorithms using smaller divisions to construct a larger
                // division, we often encounter a situation where the approximate `quo` value
                // calculated from a smaller division is multiple increments away from the true
                // `quo` value. In those algorithms, multiple correction steps have to be applied.
                // Those correction steps may need more multiplications to test `duo - (quo*div)`
                // again. Because of the fact that our `quo` can only be one of two values, we can
                // see if `duo - (quo*div)` overflows. If it did overflow, then we know that we have
                // the larger of the two values (since the true quotient is unique, and any larger
                // quotient will cause `duo - (quo*div)` to be negative). Also because there is only
                // one correction needed, we can calculate the remainder `duo - (true_quo*div) ==
                // duo - ((quo - 1)*div) == duo - (quo*div - div) == duo + div - quo*div`.
                // If `duo - (quo*div)` did not overflow, then we have the correct answer.
                let shift = n - duo_lz;
                let duo_sig_n = (duo >> shift) as $uX;
                let div_sig_n = (div >> shift) as $uX;
                let quo = $half_division(duo_sig_n, div_sig_n).0;

                // The larger `quo` value can overflow `$uD` in the right circumstances. This is a
                // manual `carrying_mul_add` with overflow checking.
                let div_lo = div as $uX;
                let div_hi = (div >> n) as $uX;
                let (tmp_lo, carry) = carrying_mul(quo, div_lo);
                let (tmp_hi, overflow) = carrying_mul_add(quo, div_hi, carry);
                let tmp = (tmp_lo as $uD) | ((tmp_hi as $uD) << n);
                if (overflow != 0) || (duo < tmp) {
                    return (
                        (quo - 1) as $uD,
                        // Both the addition and subtraction can overflow, but when combined end up
                        // as a correct positive number.
                        duo.wrapping_add(div).wrapping_sub(tmp),
                    );
                } else {
                    return (quo as $uD, duo - tmp);
                }
            }

            // Undersubtracting long division algorithm.
            // Instead of clearing a minimum of 1 bit from `duo` per iteration via binary long
            // division, `n_h - 1` bits are cleared per iteration with this algorithm. It is a more
            // complicated version of regular long division. Most integer division algorithms tend
            // to guess a part of the quotient, and may have a larger quotient than the true
            // quotient (which when multiplied by `div` will "oversubtract" the original dividend).
            // They then check if the quotient was in fact too large and then have to correct it.
            // This long division algorithm has been carefully constructed to always underguess the
            // quotient by slim margins. This allows different subalgorithms to be blindly jumped to
            // without needing an extra correction step.
            //
            // The only problem is that this subalgorithm will not work for many ranges of `duo` and
            // `div`. Fortunately, the short division, two possibility algorithm, and other simple
            // cases happen to exactly fill these gaps.
            //
            // For an example, consider the division of 76543210 by 213 and assume that `n_h` is
            // equal to two decimal digits (note: we are working with base 10 here for readability).
            // The first `sig_n_h` part of the divisor (21) is taken and is incremented by 1 to
            // prevent oversubtraction. We also record the number of extra places not a part of
            // the `sig_n` or `sig_n_h` parts.
            //
            // sig_n_h == 2 digits, sig_n == 4 digits
            //
            // vvvv     <- `duo_sig_n`
            // 76543210
            //     ^^^^ <- extra places in duo, `duo_extra == 4`
            //
            // vv  <- `div_sig_n_h`
            // 213
            //   ^ <- extra places in div, `div_extra == 1`
            //
            // The difference in extra places, `duo_extra - div_extra == extra_shl == 3`, is used
            // for shifting partial sums in the long division.
            //
            // In the first step, the first `sig_n` part of duo (7654) is divided by
            // `div_sig_n_h_add_1` (22), which results in a partial quotient of 347. This is
            // multiplied by the whole divisor to make 73911, which is shifted left by `extra_shl`
            // and subtracted from duo. The partial quotient is also shifted left by `extra_shl` to
            // be added to `quo`.
            //
            //    347
            //  ________
            // |76543210
            // -73911
            //   2632210
            //
            // Variables dependent on duo have to be updated:
            //
            // vvvv    <- `duo_sig_n == 2632`
            // 2632210
            //     ^^^ <- `duo_extra == 3`
            //
            // `extra_shl == 2`
            //
            // Two more steps are taken after this and then duo fits into `n` bits, and then a final
            // normal long division step is made. The partial quotients are all progressively added
            // to each other in the actual algorithm, but here I have left them all in a tower that
            // can be added together to produce the quotient, 359357.
            //
            //        14
            //       443
            //     119
            //    347
            //  ________
            // |76543210
            // -73911
            //   2632210
            //  -25347
            //     97510
            //    -94359
            //      3151
            //     -2982
            //       169 <- the remainder

            let mut duo = duo;
            let mut quo: $uD = 0;

            // The number of lesser significant bits not a part of `div_sig_n_h`
            let div_extra = (n + $n_h) - div_lz;

            // The most significant `n_h` bits of div
            let div_sig_n_h = (div >> div_extra) as $uH;

            // This needs to be a `$uX` in case of overflow from the increment
            let div_sig_n_h_add1 = (div_sig_n_h as $uX) + 1;

            // `{2^n, 2^(div_sb + n_h)} <= duo < 2^n_d`
            // `2^n_h <= div < {2^(duo_sb - n_h), 2^n}`
            loop {
                // The number of lesser significant bits not a part of `duo_sig_n`
                let duo_extra = n - duo_lz;

                // The most significant `n` bits of `duo`
                let duo_sig_n = (duo >> duo_extra) as $uX;

                // the two possibility algorithm requires that the difference between msbs is less
                // than `n_h`, so the comparison is `<=` here.
                if div_extra <= duo_extra {
                    // Undersubtracting long division step
                    let quo_part = $half_division(duo_sig_n, div_sig_n_h_add1).0 as $uD;
                    let extra_shl = duo_extra - div_extra;

                    // Addition to the quotient.
                    quo += (quo_part << extra_shl);

                    // Subtraction from `duo`. At least `n_h - 1` bits are cleared from `duo` here.
                    duo -= (div.wrapping_mul(quo_part) << extra_shl);
                } else {
                    // Two possibility algorithm
                    let shift = n - duo_lz;
                    let duo_sig_n = (duo >> shift) as $uX;
                    let div_sig_n = (div >> shift) as $uX;
                    let quo_part = $half_division(duo_sig_n, div_sig_n).0;
                    let div_lo = div as $uX;
                    let div_hi = (div >> n) as $uX;

                    let (tmp_lo, carry) = carrying_mul(quo_part, div_lo);
                    // The undersubtracting long division algorithm has already run once, so
                    // overflow beyond `$uD` bits is not possible here
                    let (tmp_hi, _) = carrying_mul_add(quo_part, div_hi, carry);
                    let tmp = (tmp_lo as $uD) | ((tmp_hi as $uD) << n);

                    if duo < tmp {
                        return (
                            quo + ((quo_part - 1) as $uD),
                            duo.wrapping_add(div).wrapping_sub(tmp),
                        );
                    } else {
                        return (quo + (quo_part as $uD), duo - tmp);
                    }
                }

                duo_lz = duo.leading_zeros();

                if div_lz <= duo_lz {
                    // quotient can have 0 or 1 added to it
                    if div <= duo {
                        return (quo + 1, duo - div);
                    } else {
                        return (quo, duo);
                    }
                }

                // This can only happen if `div_sd < n` (because of previous "quo = 0 or 1"
                // branches), but it is not worth it to unroll further.
                if n <= duo_lz {
                    // simple division and addition
                    let tmp = $half_division(duo as $uX, div as $uX);
                    return (quo + (tmp.0 as $uD), tmp.1 as $uD);
                }
            }
        }
    };
}
