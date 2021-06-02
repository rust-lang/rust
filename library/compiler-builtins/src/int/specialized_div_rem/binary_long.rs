/// Creates an unsigned division function that uses binary long division, designed for
/// computer architectures without division instructions. These functions have good performance for
/// microarchitectures with large branch miss penalties and architectures without the ability to
/// predicate instructions. For architectures with predicated instructions, one of the algorithms
/// described in the documentation of these functions probably has higher performance, and a custom
/// assembly routine should be used instead.
#[allow(unused_macros)]
macro_rules! impl_binary_long {
    (
        $fn:ident, // name of the unsigned division function
        $zero_div_fn:ident, // function called when division by zero is attempted
        $normalization_shift:ident, // function for finding the normalization shift
        $n:tt, // the number of bits in a $iX or $uX
        $uX:ident, // unsigned integer type for the inputs and outputs of `$fn`
        $iX:ident // signed integer type with same bitwidth as `$uX`
    ) => {
        /// Computes the quotient and remainder of `duo` divided by `div` and returns them as a
        /// tuple.
        pub fn $fn(duo: $uX, div: $uX) -> ($uX, $uX) {
            let mut duo = duo;
            // handle edge cases before calling `$normalization_shift`
            if div == 0 {
                $zero_div_fn()
            }
            if duo < div {
                return (0, duo);
            }

            // There are many variations of binary division algorithm that could be used. This
            // documentation gives a tour of different methods so that future readers wanting to
            // optimize further do not have to painstakingly derive them. The SWAR variation is
            // especially hard to understand without reading the less convoluted methods first.

            // You may notice that a `duo < div_original` check is included in many these
            // algorithms. A critical optimization that many algorithms miss is handling of
            // quotients that will turn out to have many trailing zeros or many leading zeros. This
            // happens in cases of exact or close-to-exact divisions, divisions by power of two, and
            // in cases where the quotient is small. The `duo < div_original` check handles these
            // cases of early returns and ends up replacing other kinds of mundane checks that
            // normally terminate a binary division algorithm.
            //
            // Something you may see in other algorithms that is not special-cased here is checks
            // for division by powers of two. The `duo < div_original` check handles this case and
            // more, however it can be checked up front before the bisection using the
            // `((div > 0) && ((div & (div - 1)) == 0))` trick. This is not special-cased because
            // compilers should handle most cases where divisions by power of two occur, and we do
            // not want to add on a few cycles for every division operation just to save a few
            // cycles rarely.

            // The following example is the most straightforward translation from the way binary
            // long division is typically visualized:
            // Dividing 178u8 (0b10110010) by 6u8 (0b110). `div` is shifted left by 5, according to
            // the result from `$normalization_shift(duo, div, false)`.
            //
            // Step 0: `sub` is negative, so there is not full normalization, so no `quo` bit is set
            // and `duo` is kept unchanged.
            // duo:10110010, div_shifted:11000000, sub:11110010, quo:00000000, shl:5
            //
            // Step 1: `sub` is positive, set a `quo` bit and update `duo` for next step.
            // duo:10110010, div_shifted:01100000, sub:01010010, quo:00010000, shl:4
            //
            // Step 2: Continue based on `sub`. The `quo` bits start accumulating.
            // duo:01010010, div_shifted:00110000, sub:00100010, quo:00011000, shl:3
            // duo:00100010, div_shifted:00011000, sub:00001010, quo:00011100, shl:2
            // duo:00001010, div_shifted:00001100, sub:11111110, quo:00011100, shl:1
            // duo:00001010, div_shifted:00000110, sub:00000100, quo:00011100, shl:0
            // The `duo < div_original` check terminates the algorithm with the correct quotient of
            // 29u8 and remainder of 4u8
            /*
            let div_original = div;
            let mut shl = $normalization_shift(duo, div, false);
            let mut quo = 0;
            loop {
                let div_shifted = div << shl;
                let sub = duo.wrapping_sub(div_shifted);
                // it is recommended to use `println!`s like this if functionality is unclear
                /*
                println!("duo:{:08b}, div_shifted:{:08b}, sub:{:08b}, quo:{:08b}, shl:{}",
                    duo,
                    div_shifted,
                    sub,
                    quo,
                    shl
                );
                */
                if 0 <= (sub as $iX) {
                    duo = sub;
                    quo += 1 << shl;
                    if duo < div_original {
                        // this branch is optional
                        return (quo, duo)
                    }
                }
                if shl == 0 {
                    return (quo, duo)
                }
                shl -= 1;
            }
            */

            // This restoring binary long division algorithm reduces the number of operations
            // overall via:
            // - `pow` can be shifted right instead of recalculating from `shl`
            // - starting `div` shifted left and shifting it right for each step instead of
            //   recalculating from `shl`
            // - The `duo < div_original` branch is used to terminate the algorithm instead of the
            //   `shl == 0` branch. This check is strong enough to prevent set bits of `pow` and
            //   `div` from being shifted off the end. This check also only occurs on half of steps
            //   on average, since it is behind the `(sub as $iX) >= 0` branch.
            // - `shl` is now not needed by any aspect of of the loop and thus only 3 variables are
            //   being updated between steps
            //
            // There are many variations of this algorithm, but this encompases the largest number
            // of architectures and does not rely on carry flags, add-with-carry, or SWAR
            // complications to be decently fast.
            /*
            let div_original = div;
            let shl = $normalization_shift(duo, div, false);
            let mut div: $uX = div << shl;
            let mut pow: $uX = 1 << shl;
            let mut quo: $uX = 0;
            loop {
                let sub = duo.wrapping_sub(div);
                if 0 <= (sub as $iX) {
                    duo = sub;
                    quo |= pow;
                    if duo < div_original {
                        return (quo, duo)
                    }
                }
                div >>= 1;
                pow >>= 1;
            }
            */

            // If the architecture has flags and predicated arithmetic instructions, it is possible
            // to do binary long division without branching and in only 3 or 4 instructions. This is
            // a variation of a 3 instruction central loop from
            // http://www.chiark.greenend.org.uk/~theom/riscos/docs/ultimate/a252div.txt.
            //
            // What allows doing division in only 3 instructions is realizing that instead of
            // keeping `duo` in place and shifting `div` right to align bits, `div` can be kept in
            // place and `duo` can be shifted left. This means `div` does not have to be updated,
            // but causes edge case problems and makes `duo < div_original` tests harder. Some
            // architectures have an option to shift an argument in an arithmetic operation, which
            // means `duo` can be shifted left and subtracted from in one instruction. The other two
            // instructions are updating `quo` and undoing the subtraction if it turns out things
            // were not normalized.

            /*
            // Perform one binary long division step on the already normalized arguments, because
            // the main. Note that this does a full normalization since the central loop needs
            // `duo.leading_zeros()` to be at least 1 more than `div.leading_zeros()`. The original
            // variation only did normalization to the nearest 4 steps, but this makes handling edge
            // cases much harder. We do a full normalization and perform a binary long division
            // step. In the edge case where the msbs of `duo` and `div` are set, it clears the msb
            // of `duo`, then the edge case handler shifts `div` right and does another long
            // division step to always insure `duo.leading_zeros() + 1 >= div.leading_zeros()`.
            let div_original = div;
            let mut shl = $normalization_shift(duo, div, true);
            let mut div: $uX = (div << shl);
            let mut quo: $uX = 1;
            duo = duo.wrapping_sub(div);
            if duo < div_original {
                return (1 << shl, duo);
            }
            let div_neg: $uX;
            if (div as $iX) < 0 {
                // A very ugly edge case where the most significant bit of `div` is set (after
                // shifting to match `duo` when its most significant bit is at the sign bit), which
                // leads to the sign bit of `div_neg` being cut off and carries not happening when
                // they should. This branch performs a long division step that keeps `duo` in place
                // and shifts `div` down.
                div >>= 1;
                div_neg = div.wrapping_neg();
                let (sub, carry) = duo.overflowing_add(div_neg);
                duo = sub;
                quo = quo.wrapping_add(quo).wrapping_add(carry as $uX);
                if !carry {
                    duo = duo.wrapping_add(div);
                }
                shl -= 1;
            } else {
                div_neg = div.wrapping_neg();
            }
            // The add-with-carry that updates `quo` needs to have the carry set when a normalized
            // subtract happens. Using `duo.wrapping_shl(1).overflowing_sub(div)` to do the
            // subtraction generates a carry when an unnormalized subtract happens, which is the
            // opposite of what we want. Instead, we use
            // `duo.wrapping_shl(1).overflowing_add(div_neg)`, where `div_neg` is negative `div`.
            let mut i = shl;
            loop {
                if i == 0 {
                    break;
                }
                i -= 1;
                // `ADDS duo, div, duo, LSL #1`
                // (add `div` to `duo << 1` and set flags)
                let (sub, carry) = duo.wrapping_shl(1).overflowing_add(div_neg);
                duo = sub;
                // `ADC quo, quo, quo`
                // (add with carry). Effectively shifts `quo` left by 1 and sets the least
                // significant bit to the carry.
                quo = quo.wrapping_add(quo).wrapping_add(carry as $uX);
                // `ADDCC duo, duo, div`
                // (add if carry clear). Undoes the subtraction if no carry was generated.
                if !carry {
                    duo = duo.wrapping_add(div);
                }
            }
            return (quo, duo >> shl);
            */

            // This is the SWAR (SIMD within in a register) restoring division algorithm.
            // This combines several ideas of the above algorithms:
            //  - If `duo` is shifted left instead of shifting `div` right like in the 3 instruction
            //    restoring division algorithm, some architectures can do the shifting and
            //    subtraction step in one instruction.
            //  - `quo` can be constructed by adding powers-of-two to it or shifting it left by one
            //    and adding one.
            //  - Every time `duo` is shifted left, there is another unused 0 bit shifted into the
            //    LSB, so what if we use those bits to store `quo`?
            // Through a complex setup, it is possible to manage `duo` and `quo` in the same
            // register, and perform one step with 2 or 3 instructions. The only major downsides are
            // that there is significant setup (it is only saves instructions if `shl` is
            // approximately more than 4), `duo < div_original` checks are impractical once SWAR is
            // initiated, and the number of division steps taken has to be exact (we cannot do more
            // division steps than `shl`, because it introduces edge cases where quotient bits in
            // `duo` start to collide with the real part of `div`.
            /*
            // first step. The quotient bit is stored in `quo` for now
            let div_original = div;
            let mut shl = $normalization_shift(duo, div, true);
            let mut div: $uX = (div << shl);
            duo = duo.wrapping_sub(div);
            let mut quo: $uX = 1 << shl;
            if duo < div_original {
                return (quo, duo);
            }

            let mask: $uX;
            if (div as $iX) < 0 {
                // deal with same edge case as the 3 instruction restoring division algorithm, but
                // the quotient bit from this step also has to be stored in `quo`
                div >>= 1;
                shl -= 1;
                let tmp = 1 << shl;
                mask = tmp - 1;
                let sub = duo.wrapping_sub(div);
                if (sub as $iX) >= 0 {
                    // restore
                    duo = sub;
                    quo |= tmp;
                }
                if duo < div_original {
                    return (quo, duo);
                }
            } else {
                mask = quo - 1;
            }
            // There is now room for quotient bits in `duo`.

            // Note that `div` is already shifted left and has `shl` unset bits. We subtract 1 from
            // `div` and end up with the subset of `shl` bits being all being set. This subset acts
            // just like a two's complement negative one. The subset of `div` containing the divisor
            // had 1 subtracted from it, but a carry will always be generated from the `shl` subset
            // as long as the quotient stays positive.
            //
            // When the modified `div` is subtracted from `duo.wrapping_shl(1)`, the `shl` subset
            // adds a quotient bit to the least significant bit.
            // For example, 89 (0b01011001) divided by 3 (0b11):
            //
            // shl:4, div:0b00110000
            // first step:
            //       duo:0b01011001
            // + div_neg:0b11010000
            // ____________________
            //           0b00101001
            // quo is set to 0b00010000 and mask is set to 0b00001111 for later
            //
            // 1 is subtracted from `div`. I will differentiate the `shl` part of `div` and the
            // quotient part of `duo` with `^`s.
            // chars.
            //     div:0b00110000
            //               ^^^^
            //   +     0b11111111
            //   ________________
            //         0b00101111
            //               ^^^^
            // div_neg:0b11010001
            //
            // first SWAR step:
            //  duo_shl1:0b01010010
            //                    ^
            // + div_neg:0b11010001
            // ____________________
            //           0b00100011
            //                    ^
            // second:
            //  duo_shl1:0b01000110
            //                   ^^
            // + div_neg:0b11010001
            // ____________________
            //           0b00010111
            //                   ^^
            // third:
            //  duo_shl1:0b00101110
            //                  ^^^
            // + div_neg:0b11010001
            // ____________________
            //           0b11111111
            //                  ^^^
            // 3 steps resulted in the quotient with 3 set bits as expected, but currently the real
            // part of `duo` is negative and the third step was an unnormalized step. The restore
            // branch then restores `duo`. Note that the restore branch does not shift `duo` left.
            //
            //   duo:0b11111111
            //              ^^^
            // + div:0b00101111
            //             ^^^^
            // ________________
            //       0b00101110
            //              ^^^
            // `duo` is now back in the `duo_shl1` state it was at in the the third step, with an
            // unset quotient bit.
            //
            // final step (`shl` was 4, so exactly 4 steps must be taken)
            //  duo_shl1:0b01011100
            //                 ^^^^
            // + div_neg:0b11010001
            // ____________________
            //           0b00101101
            //                 ^^^^
            // The quotient includes the `^` bits added with the `quo` bits from the beginning that
            // contained the first step and potential edge case step,
            // `quo:0b00010000 + (duo:0b00101101 & mask:0b00001111) == 0b00011101 == 29u8`.
            // The remainder is the bits remaining in `duo` that are not part of the quotient bits,
            // `duo:0b00101101 >> shl == 0b0010 == 2u8`.
            let div: $uX = div.wrapping_sub(1);
            let mut i = shl;
            loop {
                if i == 0 {
                    break;
                }
                i -= 1;
                duo = duo.wrapping_shl(1).wrapping_sub(div);
                if (duo as $iX) < 0 {
                    // restore
                    duo = duo.wrapping_add(div);
                }
            }
            // unpack the results of SWAR
            return ((duo & mask) | quo, duo >> shl);
            */

            // The problem with the conditional restoring SWAR algorithm above is that, in practice,
            // it requires assembly code to bring out its full unrolled potential (It seems that
            // LLVM can't use unrolled conditionals optimally and ends up erasing all the benefit
            // that my algorithm intends. On architectures without predicated instructions, the code
            // gen is especially bad. We need a default software division algorithm that is
            // guaranteed to get decent code gen for the central loop.

            // For non-SWAR algorithms, there is a way to do binary long division without
            // predication or even branching. This involves creating a mask from the sign bit and
            // performing different kinds of steps using that.
            /*
            let shl = $normalization_shift(duo, div, true);
            let mut div: $uX = div << shl;
            let mut pow: $uX = 1 << shl;
            let mut quo: $uX = 0;
            loop {
                let sub = duo.wrapping_sub(div);
                let sign_mask = !((sub as $iX).wrapping_shr($n - 1) as $uX);
                duo -= div & sign_mask;
                quo |= pow & sign_mask;
                div >>= 1;
                pow >>= 1;
                if pow == 0 {
                    break;
                }
            }
            return (quo, duo);
            */
            // However, it requires about 4 extra operations (smearing the sign bit, negating the
            // mask, and applying the mask twice) on top of the operations done by the actual
            // algorithm. With SWAR however, just 2 extra operations are needed, making it
            // practical and even the most optimal algorithm for some architectures.

            // What we do is use custom assembly for predicated architectures that need software
            // division, and for the default algorithm use a mask based restoring SWAR algorithm
            // without conditionals or branches. On almost all architectures, this Rust code is
            // guaranteed to compile down to 5 assembly instructions or less for each step, and LLVM
            // will unroll it in a decent way.

            // standard opening for SWAR algorithm with first step and edge case handling
            let div_original = div;
            let mut shl = $normalization_shift(duo, div, true);
            let mut div: $uX = (div << shl);
            duo = duo.wrapping_sub(div);
            let mut quo: $uX = 1 << shl;
            if duo < div_original {
                return (quo, duo);
            }
            let mask: $uX;
            if (div as $iX) < 0 {
                div >>= 1;
                shl -= 1;
                let tmp = 1 << shl;
                mask = tmp - 1;
                let sub = duo.wrapping_sub(div);
                if (sub as $iX) >= 0 {
                    duo = sub;
                    quo |= tmp;
                }
                if duo < div_original {
                    return (quo, duo);
                }
            } else {
                mask = quo - 1;
            }

            // central loop
            div = div.wrapping_sub(1);
            let mut i = shl;
            loop {
                if i == 0 {
                    break;
                }
                i -= 1;
                // shift left 1 and subtract
                duo = duo.wrapping_shl(1).wrapping_sub(div);
                // create mask
                let mask = (duo as $iX).wrapping_shr($n - 1) as $uX;
                // restore
                duo = duo.wrapping_add(div & mask);
            }
            // unpack
            return ((duo & mask) | quo, duo >> shl);

            // miscellanious binary long division algorithms that might be better for specific
            // architectures

            // Another kind of long division uses an interesting fact that `div` and `pow` can be
            // negated when `duo` is negative to perform a "negated" division step that works in
            // place of any normalization mechanism. This is a non-restoring division algorithm that
            // is very similar to the non-restoring division algorithms that can be found on the
            // internet, except there is only one test for `duo < 0`. The subtraction from `quo` can
            // be viewed as shifting the least significant set bit right (e.x. if we enter a series
            // of negated binary long division steps starting with `quo == 0b1011_0000` and
            // `pow == 0b0000_1000`, `quo` will progress like this: 0b1010_1000, 0b1010_0100,
            // 0b1010_0010, 0b1010_0001).
            /*
            let div_original = div;
            let shl = $normalization_shift(duo, div, true);
            let mut div: $uX = (div << shl);
            let mut pow: $uX = 1 << shl;
            let mut quo: $uX = pow;
            duo = duo.wrapping_sub(div);
            if duo < div_original {
                return (quo, duo);
            }
            div >>= 1;
            pow >>= 1;
            loop {
                if (duo as $iX) < 0 {
                    // Negated binary long division step.
                    duo = duo.wrapping_add(div);
                    quo = quo.wrapping_sub(pow);
                } else {
                    // Normal long division step.
                    if duo < div_original {
                        return (quo, duo)
                    }
                    duo = duo.wrapping_sub(div);
                    quo = quo.wrapping_add(pow);
                }
                pow >>= 1;
                div >>= 1;
            }
            */

            // This is the Nonrestoring SWAR algorithm, combining the nonrestoring algorithm with
            // SWAR techniques that makes the only difference between steps be negation of `div`.
            // If there was an architecture with an instruction that negated inputs to an adder
            // based on conditionals, and in place shifting (or a three input addition operation
            // that can have `duo` as two of the inputs to effectively shift it left by 1), then a
            // single instruction central loop is possible. Microarchitectures often have inputs to
            // their ALU that can invert the arguments and carry in of adders, but the architectures
            // unfortunately do not have an instruction to dynamically invert this input based on
            // conditionals.
            /*
            // SWAR opening
            let div_original = div;
            let mut shl = $normalization_shift(duo, div, true);
            let mut div: $uX = (div << shl);
            duo = duo.wrapping_sub(div);
            let mut quo: $uX = 1 << shl;
            if duo < div_original {
                return (quo, duo);
            }
            let mask: $uX;
            if (div as $iX) < 0 {
                div >>= 1;
                shl -= 1;
                let tmp = 1 << shl;
                let sub = duo.wrapping_sub(div);
                if (sub as $iX) >= 0 {
                    // restore
                    duo = sub;
                    quo |= tmp;
                }
                if duo < div_original {
                    return (quo, duo);
                }
                mask = tmp - 1;
            } else {
                mask = quo - 1;
            }

            // central loop
            let div: $uX = div.wrapping_sub(1);
            let mut i = shl;
            loop {
                if i == 0 {
                    break;
                }
                i -= 1;
                // note: the `wrapping_shl(1)` can be factored out, but would require another
                // restoring division step to prevent `(duo as $iX)` from overflowing
                if (duo as $iX) < 0 {
                    // Negated binary long division step.
                    duo = duo.wrapping_shl(1).wrapping_add(div);
                } else {
                    // Normal long division step.
                    duo = duo.wrapping_shl(1).wrapping_sub(div);
                }
            }
            if (duo as $iX) < 0 {
                // Restore. This was not needed in the original nonrestoring algorithm because of
                // the `duo < div_original` checks.
                duo = duo.wrapping_add(div);
            }
            // unpack
            return ((duo & mask) | quo, duo >> shl);
            */
        }
    };
}
