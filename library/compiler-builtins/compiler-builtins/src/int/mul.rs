use crate::support::{DInt, HInt, Int};

trait Mul: DInt + Int
where
    Self::H: DInt,
{
    fn mul(self, rhs: Self) -> Self {
        // In order to prevent infinite recursion, we cannot use the `widen_mul` in this:
        //self.lo().widen_mul(rhs.lo())
        //    .wrapping_add(self.lo().wrapping_mul(rhs.hi()).widen_hi())
        //    .wrapping_add(self.hi().wrapping_mul(rhs.lo()).widen_hi())

        let lhs_lo = self.lo();
        let rhs_lo = rhs.lo();
        // construct the widening multiplication using only `Self::H` sized multiplications
        let tmp_0 = lhs_lo.lo().zero_widen_mul(rhs_lo.lo());
        let tmp_1 = lhs_lo.lo().zero_widen_mul(rhs_lo.hi());
        let tmp_2 = lhs_lo.hi().zero_widen_mul(rhs_lo.lo());
        let tmp_3 = lhs_lo.hi().zero_widen_mul(rhs_lo.hi());
        // sum up all widening partials
        let mul = Self::from_lo_hi(tmp_0, tmp_3)
            .wrapping_add(tmp_1.zero_widen() << (Self::BITS / 4))
            .wrapping_add(tmp_2.zero_widen() << (Self::BITS / 4));
        // add the higher partials
        mul.wrapping_add(lhs_lo.wrapping_mul(rhs.hi()).widen_hi())
            .wrapping_add(self.hi().wrapping_mul(rhs_lo).widen_hi())
    }
}

impl Mul for u64 {}
impl Mul for i128 {}

pub(crate) trait UMulo: DInt + Int {
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
    // Ancient Egyptian/Ethiopian/Russian multiplication method
    // see https://en.wikipedia.org/wiki/Ancient_Egyptian_multiplication
    //
    // This is a long-available stock algorithm; e.g. it is documented in
    // Knuth's "The Art of Computer Programming" volume 2 (under the section
    // "Evaluation of Powers") since at least the 2nd edition (1981).
    //
    // The main attraction of this method is that it implements (software)
    // multiplication atop four simple operations: doubling, halving, checking
    // if a value is even/odd, and addition. This is *not* considered to be the
    // fastest multiplication method, but it may be amongst the simplest (and
    // smallest with respect to code size).
    //
    // for reference, see also implementation from gcc
    // https://raw.githubusercontent.com/gcc-mirror/gcc/master/libgcc/config/epiphany/mulsi3.c
    //
    // and from LLVM (in relatively readable RISC-V assembly):
    // https://github.com/llvm/llvm-project/blob/main/compiler-rt/lib/builtins/riscv/int_mul_impl.inc
    #[cfg(any(target_arch = "riscv32", target_arch = "riscv64", target_arch = "m68k"))]
    pub extern "C" fn __mulsi3(a: u32, b: u32) -> u32 {
        let (mut a, mut b) = (a, b);
        let mut r: u32 = 0;

        while a > 0 {
            if a & 1 > 0 {
                r = r.wrapping_add(b);
            }
            a >>= 1;
            b <<= 1;
        }

        r
    }

    #[maybe_use_optimized_c_shim]
    #[arm_aeabi_alias = __aeabi_lmul]
    pub extern "C" fn __muldi3(a: u64, b: u64) -> u64 {
        #[cfg(all(any(target_arch = "riscv32", target_arch = "riscv64"), not(target_feature = "m")))]
        {
            let (mut a, mut b) = (a, b);
            let mut r: u64 = 0;

            while a > 0 {
                if a & 1 > 0 {
                    r = r.wrapping_add(b);
                }
                a >>= 1;
                b <<= 1;
            }

            r
        }

        #[cfg(not(all(any(target_arch = "riscv32", target_arch = "riscv64"), not(target_feature = "m"))))]
        {
            a.mul(b)
        }
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

    pub extern "C" fn __rust_i128_mulo(a: i128, b: i128, oflow: &mut i32) -> i128 {
        let (mul, o) = i128_overflowing_mul(a, b);
        *oflow = o.into();
        mul
    }

    pub extern "C" fn __rust_u128_mulo(a: u128, b: u128, oflow: &mut i32) -> u128 {
        let (mul, o) = a.mulo(b);
        *oflow = o.into();
        mul
    }

}
