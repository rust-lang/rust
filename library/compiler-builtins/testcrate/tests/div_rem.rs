use rand_xoshiro::rand_core::{RngCore, SeedableRng};
use rand_xoshiro::Xoshiro128StarStar;

use compiler_builtins::int::sdiv::{__divmoddi4, __divmodsi4, __divmodti4};
use compiler_builtins::int::udiv::{__udivmoddi4, __udivmodsi4, __udivmodti4};

/// Creates intensive test functions for division functions of a certain size
macro_rules! test {
    (
        $n:expr, // the number of bits in a $iX or $uX
        $uX:ident, // unsigned integer that will be shifted
        $iX:ident, // signed version of $uX
        $test_name:ident, // name of the test function
        $unsigned_name:ident, // unsigned division function
        $signed_name:ident // signed division function
    ) => {
        #[test]
        fn $test_name() {
            fn assert_invariants(lhs: $uX, rhs: $uX) {
                let rem: &mut $uX = &mut 0;
                let quo: $uX = $unsigned_name(lhs, rhs, Some(rem));
                let rem = *rem;
                if rhs <= rem || (lhs != rhs.wrapping_mul(quo).wrapping_add(rem)) {
                    panic!(
                        "unsigned division function failed with lhs:{} rhs:{} \
                        expected:({}, {}) found:({}, {})",
                        lhs,
                        rhs,
                        lhs.wrapping_div(rhs),
                        lhs.wrapping_rem(rhs),
                        quo,
                        rem
                    );
                }

                // test the signed division function also
                let lhs = lhs as $iX;
                let rhs = rhs as $iX;
                let mut rem: $iX = 0;
                let quo: $iX = $signed_name(lhs, rhs, &mut rem);
                // We cannot just test that
                // `lhs == rhs.wrapping_mul(quo).wrapping_add(rem)`, but also
                // need to make sure the remainder isn't larger than the divisor
                // and has the correct sign.
                let incorrect_rem = if rem == 0 {
                    false
                } else if rhs == $iX::MIN {
                    // `rhs.wrapping_abs()` would overflow, so handle this case
                    // separately.
                    (lhs.is_negative() != rem.is_negative()) || (rem == $iX::MIN)
                } else {
                    (lhs.is_negative() != rem.is_negative())
                        || (rhs.wrapping_abs() <= rem.wrapping_abs())
                };
                if incorrect_rem || lhs != rhs.wrapping_mul(quo).wrapping_add(rem) {
                    panic!(
                        "signed division function failed with lhs:{} rhs:{} \
                        expected:({}, {}) found:({}, {})",
                        lhs,
                        rhs,
                        lhs.wrapping_div(rhs),
                        lhs.wrapping_rem(rhs),
                        quo,
                        rem
                    );
                }
            }

            // Specially designed random fuzzer
            let mut rng = Xoshiro128StarStar::seed_from_u64(0);
            let mut lhs: $uX = 0;
            let mut rhs: $uX = 0;
            // all ones constant
            let ones: $uX = !0;
            // Alternating ones and zeros (e.x. 0b1010101010101010). This catches second-order
            // problems that might occur for algorithms with two modes of operation (potentially
            // there is some invariant that can be broken for large `duo` and maintained via
            // alternating between modes, breaking the algorithm when it reaches the end).
            let mut alt_ones: $uX = 1;
            for _ in 0..($n / 2) {
                alt_ones <<= 2;
                alt_ones |= 1;
            }
            // creates a mask for indexing the bits of the type
            let bit_indexing_mask = $n - 1;
            for _ in 0..1_000_000 {
                // Randomly OR, AND, and XOR randomly sized and shifted continuous strings of
                // ones with `lhs` and `rhs`. This results in excellent fuzzing entropy such as:
                // lhs:10101010111101000000000100101010 rhs: 1010101010000000000000001000001
                // lhs:10101010111101000000000101001010 rhs: 1010101010101010101010100010100
                // lhs:10101010111101000000000101001010 rhs:11101010110101010101010100001110
                // lhs:10101010000000000000000001001010 rhs:10100010100000000000000000001010
                // lhs:10101010000000000000000001001010 rhs:            10101010101010101000
                // lhs:10101010000000000000000001100000 rhs:11111111111101010101010101001111
                // lhs:10101010000000101010101011000000 rhs:11111111111101010101010100000111
                // lhs:10101010101010101010101011101010 rhs:             1010100000000000000
                // lhs:11111111110101101010101011010111 rhs:             1010100000000000000
                // The msb is set half of the time by the fuzzer, but `assert_invariants` tests
                // both the signed and unsigned functions.
                let r0: u32 = bit_indexing_mask & rng.next_u32();
                let r1: u32 = bit_indexing_mask & rng.next_u32();
                let mask = ones.wrapping_shr(r0).rotate_left(r1);
                match rng.next_u32() % 8 {
                    0 => lhs |= mask,
                    1 => lhs &= mask,
                    // both 2 and 3 to make XORs as common as ORs and ANDs combined, otherwise
                    // the entropy gets destroyed too often
                    2 | 3 => lhs ^= mask,
                    4 => rhs |= mask,
                    5 => rhs &= mask,
                    _ => rhs ^= mask,
                }
                // do the same for alternating ones and zeros
                let r0: u32 = bit_indexing_mask & rng.next_u32();
                let r1: u32 = bit_indexing_mask & rng.next_u32();
                let mask = alt_ones.wrapping_shr(r0).rotate_left(r1);
                match rng.next_u32() % 8 {
                    0 => lhs |= mask,
                    1 => lhs &= mask,
                    // both 2 and 3 to make XORs as common as ORs and ANDs combined, otherwise
                    // the entropy gets destroyed too often
                    2 | 3 => lhs ^= mask,
                    4 => rhs |= mask,
                    5 => rhs &= mask,
                    _ => rhs ^= mask,
                }
                if rhs != 0 {
                    assert_invariants(lhs, rhs);
                }
            }
        }
    };
}

test!(32, u32, i32, div_rem_si4, __udivmodsi4, __divmodsi4);
test!(64, u64, i64, div_rem_di4, __udivmoddi4, __divmoddi4);
test!(128, u128, i128, div_rem_ti4, __udivmodti4, __divmodti4);
