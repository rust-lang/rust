use int::LargeInt;
use int::{DInt, HInt, Int};

trait Mul: LargeInt {
    fn mul(self, other: Self) -> Self {
        let half_bits = Self::BITS / 4;
        let lower_mask = !<<Self as LargeInt>::LowHalf>::ZERO >> half_bits;
        let mut low = (self.low() & lower_mask).wrapping_mul(other.low() & lower_mask);
        let mut t = low >> half_bits;
        low &= lower_mask;
        t += (self.low() >> half_bits).wrapping_mul(other.low() & lower_mask);
        low += (t & lower_mask) << half_bits;
        let mut high = Self::low_as_high(t >> half_bits);
        t = low >> half_bits;
        low &= lower_mask;
        t += (other.low() >> half_bits).wrapping_mul(self.low() & lower_mask);
        low += (t & lower_mask) << half_bits;
        high += Self::low_as_high(t >> half_bits);
        high += Self::low_as_high((self.low() >> half_bits).wrapping_mul(other.low() >> half_bits));
        high = high
            .wrapping_add(self.high().wrapping_mul(Self::low_as_high(other.low())))
            .wrapping_add(Self::low_as_high(self.low()).wrapping_mul(other.high()));
        Self::from_parts(low, high)
    }
}

impl Mul for u64 {}
impl Mul for i128 {}

pub(crate) trait UMulo: Int + DInt {
    fn mulo(self, rhs: Self) -> (Self, bool) {
        match (self.hi().is_zero(), rhs.hi().is_zero()) {
            // overflow is guaranteed
            (false, false) => (self.wrapping_mul(rhs), true),
            (true, false) => {
                let mul_lo = self.lo().widen_mul(rhs.lo());
                let mul_hi = self.lo().widen_mul(rhs.hi());
                let (mul, o) = mul_lo.overflowing_add(mul_hi.lo().widen_hi());
                (mul, o || !mul_hi.hi().is_zero())
            }
            (false, true) => {
                let mul_lo = rhs.lo().widen_mul(self.lo());
                let mul_hi = rhs.lo().widen_mul(self.hi());
                let (mul, o) = mul_lo.overflowing_add(mul_hi.lo().widen_hi());
                (mul, o || !mul_hi.hi().is_zero())
            }
            // overflow is guaranteed to not happen, and use a smaller widening multiplication
            (true, true) => (self.lo().widen_mul(rhs.lo()), false),
        }
    }
}

impl UMulo for u32 {}
impl UMulo for u64 {}
impl UMulo for u128 {}

macro_rules! impl_signed_mulo {
    ($fn:ident, $iD:ident, $uD:ident) => {
        fn $fn(lhs: $iD, rhs: $iD) -> ($iD, bool) {
            let mut lhs = lhs;
            let mut rhs = rhs;
            // the test against `mul_neg` below fails without this early return
            if lhs == 0 || rhs == 0 {
                return (0, false);
            }

            let lhs_neg = lhs < 0;
            let rhs_neg = rhs < 0;
            if lhs_neg {
                lhs = lhs.wrapping_neg();
            }
            if rhs_neg {
                rhs = rhs.wrapping_neg();
            }
            let mul_neg = lhs_neg != rhs_neg;

            let (mul, o) = (lhs as $uD).mulo(rhs as $uD);
            let mut mul = mul as $iD;

            if mul_neg {
                mul = mul.wrapping_neg();
            }
            if (mul < 0) != mul_neg {
                // this one check happens to catch all edge cases related to `$iD::MIN`
                (mul, true)
            } else {
                (mul, o)
            }
        }
    };
}

impl_signed_mulo!(i32_overflowing_mul, i32, u32);
impl_signed_mulo!(i64_overflowing_mul, i64, u64);
impl_signed_mulo!(i128_overflowing_mul, i128, u128);

intrinsics! {
    #[maybe_use_optimized_c_shim]
    #[arm_aeabi_alias = __aeabi_lmul]
    pub extern "C" fn __muldi3(a: u64, b: u64) -> u64 {
        a.mul(b)
    }

    pub extern "C" fn __multi3(a: i128, b: i128) -> i128 {
        a.mul(b)
    }

    pub extern "C" fn __mulosi4(a: i32, b: i32, oflow: &mut i32) -> i32 {
        let (mul, o) = i32_overflowing_mul(a, b);
        *oflow = o as i32;
        mul
    }

    pub extern "C" fn __mulodi4(a: i64, b: i64, oflow: &mut i32) -> i64 {
        let (mul, o) = i64_overflowing_mul(a, b);
        *oflow = o as i32;
        mul
    }

    #[unadjusted_on_win64]
    pub extern "C" fn __muloti4(a: i128, b: i128, oflow: &mut i32) -> i128 {
        let (mul, o) = i128_overflowing_mul(a, b);
        *oflow = o as i32;
        mul
    }

    pub extern "C" fn __rust_i128_mulo(a: i128, b: i128) -> (i128, bool) {
        i128_overflowing_mul(a, b)
    }

    pub extern "C" fn __rust_u128_mulo(a: u128, b: u128) -> (u128, bool) {
        a.mulo(b)
    }
}
