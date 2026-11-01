use libm::support::{Float, Int, MinInt};

/// Iterate all the bitwise subsets of the given mask.
///
/// Produces the same sequence as `(0..=uN::MAX).filter(|x| x & !mask == 0)`,
/// but each item is generated in O(1) time.
///
/// # Panics
///
/// Panics if the mask has all bits set.
///
/// # Example
///
/// ```ignore
/// assert!(bitwise_subsets(0b1001).eq([0b0000, 0b0001, 0b1000, 0b1001]));
/// ```
fn bitwise_subsets<I>(mask: I) -> impl Iterator<Item = I> + Clone
where
    I: Int<Unsigned = I>,
{
    assert!(
        mask != I::MAX,
        "to optimize the implementation, varying every bit is not supported"
    );
    let fixed = !mask;
    let mut counter = fixed - I::ONE;

    // `(counter + 1) & !fixed` is initially 0, and increases after each item returned
    std::iter::from_fn(move || {
        counter = counter.checked_add(I::ONE)? | fixed;
        Some(counter ^ fixed)
    })
}

/// Create a bitmask with `count` ones. The ones are filled in this order:
///
/// 1. Sign bit
/// 2. Significand low bits
/// 3. Significand high bits, exponent low bits, exponent high bits (increment together)
fn most_wanted_bitmask<F: Float>(count: u32) -> F::Int {
    if count == 0 {
        return F::Int::ZERO;
    }
    let mut mask = F::SIGN_MASK;
    let n = (count - 1) / 4;
    mask |= low_mask_bits(n, F::EXP_MASK);
    mask |= high_mask_bits(n, F::EXP_MASK);
    mask |= high_mask_bits(n, F::SIG_MASK);
    // spend most of the remaining budget on the least significant bits, then use up remainders
    mask |= low_mask_bits(count - mask.count_ones(), F::SIG_MASK);
    mask |= high_mask_bits(n + count - mask.count_ones(), F::SIG_MASK);
    mask |= high_mask_bits(n + count - mask.count_ones(), F::EXP_MASK);
    mask |= low_mask_bits(n + count - mask.count_ones(), F::EXP_MASK);
    mask
}

/// Similar to `most_wanted_bitmask` but adds bits around the middle of the int rather than at
/// a sign/exponent/significand boundary.
fn most_wanted_int_bitmask<I: Int<Unsigned = I>>(count: u32) -> I {
    if count == 0 {
        return I::ZERO;
    }
    let mut mask = I::ZERO;
    let n = count / 4;

    let low_mask = I::MAX >> (I::BITS / 2);
    let high_mask = low_mask << (I::BITS / 2);

    mask |= low_mask_bits(n, high_mask);
    mask |= high_mask_bits(n, high_mask);
    mask |= high_mask_bits(n, low_mask);
    // spend most of the remaining budget on the least significant bits, then use up remainders
    mask |= low_mask_bits(count - mask.count_ones(), low_mask);
    mask |= high_mask_bits(n + count - mask.count_ones(), low_mask);
    mask |= high_mask_bits(n + count - mask.count_ones(), high_mask);
    mask |= low_mask_bits(n + count - mask.count_ones(), high_mask);
    mask
}

/// Set `count` bits in the LSB of `mask`.
fn low_mask_bits<I: Int>(count: u32, mask: I) -> I {
    mask & (mask >> (mask.count_ones().saturating_sub(count)))
}

/// Set `count` bits in the MSB of `mask`.
fn high_mask_bits<I: Int>(count: u32, mask: I) -> I {
    mask & (mask << (mask.count_ones().saturating_sub(count)))
}

/// Biased generator for floats.
///
/// Starts with the first value of `fillers`, then toggles `bits_to_vary` bits exhaustively. Once
/// that has completed, move on to the next filler.
///
/// The returned iterator will produce `fillers.len() << bits_to_vary` items.
#[cfg_attr(not(test), expect(dead_code))]
fn float_gen<F>(
    bits_to_vary: u32,
    fillers: impl IntoIterator<Item = F::Int>,
) -> impl Iterator<Item = F>
where
    F: Float,
    F::Int: Int<Unsigned = F::Int>,
{
    let varying = most_wanted_bitmask::<F>(bits_to_vary);
    let patterns = bitwise_subsets(varying);

    fillers
        .into_iter()
        .flat_map(move |preset| patterns.clone().map(move |x| x ^ preset))
        .map(F::from_bits)
}

/// See [`float_gen`] docs for details.
#[cfg_attr(not(test), expect(dead_code))]
fn int_gen<I>(bits_to_vary: u32, fillers: impl IntoIterator<Item = I>) -> impl Iterator<Item = I>
where
    I: Int<Unsigned = I>,
{
    let varying = most_wanted_int_bitmask::<I>(bits_to_vary);
    let patterns = bitwise_subsets(varying);

    fillers
        .into_iter()
        .flat_map(move |preset| patterns.clone().map(move |x| x ^ preset))
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::f8;

    #[test]
    fn equivalence() {
        // with a small integer type, we can easily verify that behaviour matches for all arguments
        for mask in 0..u8::MAX {
            let expect = (0..=u8::MAX).filter(|x| x & !mask == 0);
            assert!(bitwise_subsets(mask).eq(expect));
        }
    }

    #[test]
    fn most_wanted() {
        let expected8 = [
            0b0_0000_000,
            //
            0b1_0000_000,
            0b1_0000_001,
            0b1_0000_011,
            0b1_0000_111,
            //
            0b1_1001_101,
            0b1_1001_111,
            0b1_1101_111,
            0b1_1111_111,
        ];
        let expected32 = [
            0b0_00000000_00000000000000000000000,
            //
            0b1_00000000_00000000000000000000000,
            0b1_00000000_00000000000000000000001,
            0b1_00000000_00000000000000000000011,
            0b1_00000000_00000000000000000000111,
            //
            0b1_10000001_10000000000000000000001,
            0b1_10000001_10000000000000000000011,
            0b1_10000001_10000000000000000000111,
            0b1_10000001_10000000000000000001111,
            //
            0b1_11000011_11000000000000000000011,
            0b1_11000011_11000000000000000000111,
            0b1_11000011_11000000000000000001111,
            0b1_11000011_11000000000000000011111,
            //
            0b1_11100111_11100000000000000000111,
            0b1_11100111_11100000000000000001111,
            0b1_11100111_11100000000000000011111,
            0b1_11100111_11100000000000000111111,
            //
            0b1_11111111_11110000000000000001111,
            0b1_11111111_11110000000000000011111,
            0b1_11111111_11110000000000000111111,
            0b1_11111111_11110000000000001111111,
            //
            0b1_11111111_11111000000000001111111,
            0b1_11111111_11111000000000011111111,
            0b1_11111111_11111000000000111111111,
            0b1_11111111_11111000000001111111111,
            //
            0b1_11111111_11111100000001111111111,
            0b1_11111111_11111100000011111111111,
            0b1_11111111_11111100000111111111111,
            0b1_11111111_11111100001111111111111,
            //
            0b1_11111111_11111110001111111111111,
            0b1_11111111_11111110011111111111111,
            0b1_11111111_11111110111111111111111,
            0b1_11111111_11111111111111111111111,
        ];
        for k in 0..=8 {
            let mask = most_wanted_bitmask::<f8>(k);
            assert_eq!(mask, expected8[k as usize], "{k}");
            assert_eq!(mask.count_ones(), k);
        }
        for k in 0..=32 {
            let mask = most_wanted_bitmask::<f32>(k);
            assert_eq!(mask, expected32[k as usize], "{k}");
            assert_eq!(mask.count_ones(), k);
        }
    }

    #[test]
    fn most_wanted_int() {
        let expected = [
            0b0000_0000_0000_0000,
            0b0000_0000_0000_0001,
            0b0000_0000_0000_0011,
            0b0000_0000_0000_0111,
            //
            0b1000_0001_1000_0001,
            0b1000_0001_1000_0011,
            0b1000_0001_1000_0111,
            0b1000_0001_1000_1111,
            //
            0b1100_0011_1100_0011,
            0b1100_0011_1100_0111,
            0b1100_0011_1100_1111,
            0b1100_0011_1101_1111,
            //
            0b1110_0111_1110_0111,
            0b1110_0111_1110_1111,
            0b1110_0111_1111_1111,
            0b1111_0111_1111_1111,
            //
            0b1111_1111_1111_1111,
        ];
        let mut x = vec![];
        for k in 0..=16 {
            let mask = most_wanted_int_bitmask::<u16>(k);
            x.push(mask);
            assert_eq!(mask, expected[k as usize], "{k}");
            assert_eq!(mask.count_ones(), k);
        }
    }

    #[test]
    #[cfg(f16_enabled)]
    fn gen_with_fillers() {
        let expected = [
            // Filler: zeros
            0b0_00000_0000000000,
            0b0_00000_0000000001,
            0b0_00000_0000000010,
            0b0_00000_0000000011,
            0b1_00000_0000000000,
            0b1_00000_0000000001,
            0b1_00000_0000000010,
            0b1_00000_0000000011,
            // Filler: ones
            0b1_11111_1111111111,
            0b1_11111_1111111110,
            0b1_11111_1111111101,
            0b1_11111_1111111100,
            0b0_11111_1111111111,
            0b0_11111_1111111110,
            0b0_11111_1111111101,
            0b0_11111_1111111100,
            // Filler: pattern
            0b1_00110_1000101011,
            0b1_00110_1000101010,
            0b1_00110_1000101001,
            0b1_00110_1000101000,
            0b0_00110_1000101011,
            0b0_00110_1000101010,
            0b0_00110_1000101001,
            0b0_00110_1000101000,
        ];
        let v = float_gen::<f16>(3, [0, u16::MAX, 0b1_00110_1000101011])
            .map(|x| x.to_bits())
            .collect::<Vec<_>>();
        assert_eq!(expected.as_slice(), v);
    }

    #[test]
    fn gen_int_with_fillers() {
        let expected = [
            // Filler: zeros
            0b0000_0000_0000_0000,
            0b0000_0000_0000_0001,
            0b0000_0000_0000_0010,
            0b0000_0000_0000_0011,
            0b0000_0000_0000_0100,
            0b0000_0000_0000_0101,
            0b0000_0000_0000_0110,
            0b0000_0000_0000_0111,
            // Filler: ones
            0b1111_1111_1111_1111,
            0b1111_1111_1111_1110,
            0b1111_1111_1111_1101,
            0b1111_1111_1111_1100,
            0b1111_1111_1111_1011,
            0b1111_1111_1111_1010,
            0b1111_1111_1111_1001,
            0b1111_1111_1111_1000,
            // Filler: pattern
            0b1001_1010_0010_1011,
            0b1001_1010_0010_1010,
            0b1001_1010_0010_1001,
            0b1001_1010_0010_1000,
            0b1001_1010_0010_1111,
            0b1001_1010_0010_1110,
            0b1001_1010_0010_1101,
            0b1001_1010_0010_1100,
        ];
        let v = int_gen::<u16>(3, [0, u16::MAX, 0b1001_1010_0010_1011]).collect::<Vec<_>>();
        assert_eq!(expected.as_slice(), v);
    }

    #[test]
    fn gen_includes_specials() {
        let v: Vec<_> = float_gen(5, vec![0, 0x7fffff, !0x7fffff, !0])
            .map(f32::to_bits)
            .collect();
        for x in &[
            0.0,
            f32::from_bits(1),
            f32::MIN_POSITIVE,
            f32::MAX,
            f32::INFINITY,
            f32::NAN,
        ] {
            assert!(v.contains(&x.to_bits()), "{x} not found");
            assert!(v.contains(&(-x).to_bits()), "-{x} not found");
        }
    }

    #[test]
    fn count() {
        for k in 0..10 {
            assert!(float_gen::<f32>(k, vec![0]).count() == 1 << k);
            assert!(float_gen::<f32>(k, vec![0, !0]).count() == 2 << k);
            assert!(float_gen::<f32>(k, vec![0, 1, 2]).count() == 3 << k);
        }
    }

    #[test]
    fn specific() {
        let iter = float_gen::<f32>(1, vec![0]).map(f32::to_bits);
        assert!(iter.eq([0.0_f32.to_bits(), (-0.0_f32).to_bits()]));
        let iter = float_gen::<f64>(1, vec![0]).map(f64::to_bits);
        assert!(iter.eq([0.0_f64.to_bits(), (-0.0_f64).to_bits()]));

        let mut v: Vec<_> = float_gen::<f32>(5, vec![0]).map(f32::to_bits).collect();
        assert!(v.len() == 32);
        assert!(v.is_sorted());
        v.dedup();
        assert!(v.len() == 32);
        for bits in v {
            assert!(bits & 0xc0c0_0001 == bits);
        }

        let mut v: Vec<_> = float_gen::<f64>(5, vec![0]).map(f64::to_bits).collect();
        assert!(v.len() == 32);
        assert!(v.is_sorted());
        v.dedup();
        assert!(v.len() == 32);
        for bits in v {
            assert!(bits & 0xc018_0000_0000_0001 == bits);
        }
    }
}
