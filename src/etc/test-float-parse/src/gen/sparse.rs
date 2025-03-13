use std::fmt::Write;

use crate::traits::BoxGenIter;
use crate::{Float, Generator};

const POWERS_OF_TWO: [u128; 128] = make_powers_of_two();

const fn make_powers_of_two() -> [u128; 128] {
    let mut ret = [0; 128];
    let mut i = 0;
    while i < 128 {
        ret[i] = 1 << i;
        i += 1;
    }

    ret
}

/// Can't clone this result because of lifetime errors, just use a macro.
macro_rules! pow_iter {
    () => {
        (0..F::BITS).map(|i| F::Int::try_from(POWERS_OF_TWO[i as usize]).unwrap())
    };
}

/// Test all numbers that include three 1s in the binary representation as integers.
pub struct FewOnesInt<F: Float>
where
    FewOnesInt<F>: Generator<F>,
{
    iter: BoxGenIter<Self, F>,
}

impl<F: Float> Generator<F> for FewOnesInt<F>
where
    <F::Int as TryFrom<u128>>::Error: std::fmt::Debug,
{
    const SHORT_NAME: &'static str = "few ones int";

    type WriteCtx = F::Int;

    fn total_tests() -> u64 {
        u64::from(F::BITS).pow(3)
    }

    fn new() -> Self {
        let iter = pow_iter!()
            .flat_map(move |a| pow_iter!().map(move |b| (a, b)))
            .flat_map(move |(a, b)| pow_iter!().map(move |c| (a, b, c)))
            .map(|(a, b, c)| a | b | c);

        Self { iter: Box::new(iter) }
    }

    fn write_string(s: &mut String, ctx: Self::WriteCtx) {
        write!(s, "{ctx}").unwrap();
    }
}

impl<F: Float> Iterator for FewOnesInt<F> {
    type Item = F::Int;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}

/// Similar to `FewOnesInt` except test those bit patterns as a float.
pub struct FewOnesFloat<F: Float>(FewOnesInt<F>);

impl<F: Float> Generator<F> for FewOnesFloat<F>
where
    <F::Int as TryFrom<u128>>::Error: std::fmt::Debug,
{
    const NAME: &'static str = "few ones float";
    const SHORT_NAME: &'static str = "few ones float";

    type WriteCtx = F;

    fn total_tests() -> u64 {
        FewOnesInt::<F>::total_tests()
    }

    fn new() -> Self {
        Self(FewOnesInt::new())
    }

    fn write_string(s: &mut String, ctx: Self::WriteCtx) {
        write!(s, "{ctx:e}").unwrap();
    }
}

impl<F: Float> Iterator for FewOnesFloat<F> {
    type Item = F;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|i| F::from_bits(i))
    }
}
