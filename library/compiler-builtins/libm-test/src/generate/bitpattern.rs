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

fn low_exp_bits<F: Float>(count: u32) -> F::Int {
    F::EXP_MASK & (F::EXP_MASK >> (F::EXP_BITS.saturating_sub(count)))
}
fn high_exp_bits<F: Float>(count: u32) -> F::Int {
    F::EXP_MASK & (F::EXP_MASK << (F::EXP_BITS.saturating_sub(count)))
}
fn low_sig_bits<F: Float>(count: u32) -> F::Int {
    F::SIG_MASK & (F::SIG_MASK >> (F::SIG_BITS.saturating_sub(count)))
}
fn high_sig_bits<F: Float>(count: u32) -> F::Int {
    F::SIG_MASK & (F::SIG_MASK << (F::SIG_BITS.saturating_sub(count)))
}
fn most_wanted_bitmask<F: Float>(count: u32) -> F::Int {
    if count == 0 {
        return F::Int::ZERO;
    }
    let mut mask = F::SIGN_MASK;
    let n = (count - 1) / 4;
    mask |= low_exp_bits::<F>(n);
    mask |= high_exp_bits::<F>(n);
    mask |= high_sig_bits::<F>(n);
    // spend the remaining budget on the least significant bits
    mask |= low_sig_bits::<F>(count - mask.count_ones());
    mask
}

/// Biased generator for floats.
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

#[cfg(test)]
mod test {
    use super::{bitwise_subsets, float_gen};
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
        let expected = [
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
        for k in 0..=32 {
            assert_eq!(super::most_wanted_bitmask::<f32>(k), expected[k as usize]);
        }
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
