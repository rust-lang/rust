/// Creates unsigned and signed division functions optimized for dividing integers with the same
/// bitwidth as the largest operand in an asymmetrically sized division. For example, x86-64 has an
/// assembly instruction that can divide a 128 bit integer by a 64 bit integer if the quotient fits
/// in 64 bits. The 128 bit version of this algorithm would use that fast hardware division to
/// construct a full 128 bit by 128 bit division.
#[macro_export]
macro_rules! impl_asymmetric {
    (
        $unsigned_name:ident, // name of the unsigned division function
        $signed_name:ident, // name of the signed division function
        $zero_div_fn:ident, // function called when division by zero is attempted
        $half_division:ident, // function for division of a $uX by a $uX
        $asymmetric_division:ident, // function for division of a $uD by a $uX
        $n_h:expr, // the number of bits in a $iH or $uH
        $uH:ident, // unsigned integer with half the bit width of $uX
        $uX:ident, // unsigned integer with half the bit width of $uD
        $uD:ident, // unsigned integer type for the inputs and outputs of `$unsigned_name`
        $iD:ident, // signed integer type for the inputs and outputs of `$signed_name`
        $($unsigned_attr:meta),*; // attributes for the unsigned function
        $($signed_attr:meta),* // attributes for the signed function
    ) => {
        /// Computes the quotient and remainder of `duo` divided by `div` and returns them as a
        /// tuple.
        $(
            #[$unsigned_attr]
        )*
        pub fn $unsigned_name(duo: $uD, div: $uD) -> ($uD,$uD) {
            let n: u32 = $n_h * 2;

            let duo_lo = duo as $uX;
            let duo_hi = (duo >> n) as $uX;
            let div_lo = div as $uX;
            let div_hi = (div >> n) as $uX;
            if div_hi == 0 {
                if div_lo == 0 {
                    $zero_div_fn()
                }
                if duo_hi < div_lo {
                    // `$uD` by `$uX` division with a quotient that will fit into a `$uX`
                    let (quo, rem) = unsafe { $asymmetric_division(duo, div_lo) };
                    return (quo as $uD, rem as $uD)
                } else {
                    // Short division using the $uD by $uX division
                    let (quo_hi, rem_hi) = $half_division(duo_hi, div_lo);
                    let tmp = unsafe {
                        $asymmetric_division((duo_lo as $uD) | ((rem_hi as $uD) << n), div_lo)
                    };
                    return ((tmp.0 as $uD) | ((quo_hi as $uD) << n), tmp.1 as $uD)
                }
            }

            // This has been adapted from
            // https://www.codeproject.com/tips/785014/uint-division-modulus which was in turn
            // adapted from Hacker's Delight. This is similar to the two possibility algorithm
            // in that it uses only more significant parts of `duo` and `div` to divide a large
            // integer with a smaller division instruction.
            let div_lz = div_hi.leading_zeros();
            let div_extra = n - div_lz;
            let div_sig_n = (div >> div_extra) as $uX;
            let tmp = unsafe {
                $asymmetric_division(duo >> 1, div_sig_n)
            };

            let mut quo = tmp.0 >> ((n - 1) - div_lz);
            if quo != 0 {
                quo -= 1;
            }

            // Note that this is a full `$uD` multiplication being used here
            let mut rem = duo - (quo as $uD).wrapping_mul(div);
            if div <= rem {
                quo += 1;
                rem -= div;
            }
            return (quo as $uD, rem)
        }

        /// Computes the quotient and remainder of `duo` divided by `div` and returns them as a
        /// tuple.
        $(
            #[$signed_attr]
        )*
        pub fn $signed_name(duo: $iD, div: $iD) -> ($iD, $iD) {
            match (duo < 0, div < 0) {
                (false, false) => {
                    let t = $unsigned_name(duo as $uD, div as $uD);
                    (t.0 as $iD, t.1 as $iD)
                },
                (true, false) => {
                    let t = $unsigned_name(duo.wrapping_neg() as $uD, div as $uD);
                    ((t.0 as $iD).wrapping_neg(), (t.1 as $iD).wrapping_neg())
                },
                (false, true) => {
                    let t = $unsigned_name(duo as $uD, div.wrapping_neg() as $uD);
                    ((t.0 as $iD).wrapping_neg(), t.1 as $iD)
                },
                (true, true) => {
                    let t = $unsigned_name(duo.wrapping_neg() as $uD, div.wrapping_neg() as $uD);
                    (t.0 as $iD, (t.1 as $iD).wrapping_neg())
                },
            }
        }
    }
}
