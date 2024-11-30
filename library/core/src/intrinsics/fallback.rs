#![unstable(
    feature = "core_intrinsics_fallbacks",
    reason = "The fallbacks will never be stable, as they exist only to be called \
              by the fallback MIR, but they're exported so they can be tested on \
              platforms where the fallback MIR isn't actually used",
    issue = "none"
)]
#![allow(missing_docs)]

#[const_trait]
pub trait CarryingMulAdd: Copy + 'static {
    type Unsigned: Copy + 'static;
    fn carrying_mul_add(
        self,
        multiplicand: Self,
        addend: Self,
        carry: Self,
    ) -> (Self::Unsigned, Self);
}

macro_rules! impl_carrying_mul_add_by_widening {
    ($($t:ident $u:ident $w:ident,)+) => {$(
        #[rustc_const_unstable(feature = "core_intrinsics_fallbacks", issue = "none")]
        impl const CarryingMulAdd for $t {
            type Unsigned = $u;
            #[inline]
            fn carrying_mul_add(self, a: Self, b: Self, c: Self) -> ($u, $t) {
                let wide = (self as $w) * (a as $w) + (b as $w) + (c as $w);
                (wide as _, (wide >> Self::BITS) as _)
            }
        }
    )+};
}
impl_carrying_mul_add_by_widening! {
    u8 u8 u16,
    u16 u16 u32,
    u32 u32 u64,
    u64 u64 u128,
    usize usize UDoubleSize,
    i8 u8 i16,
    i16 u16 i32,
    i32 u32 i64,
    i64 u64 i128,
    isize usize UDoubleSize,
}

#[cfg(target_pointer_width = "16")]
type UDoubleSize = u32;
#[cfg(target_pointer_width = "32")]
type UDoubleSize = u64;
#[cfg(target_pointer_width = "64")]
type UDoubleSize = u128;

#[inline]
const fn wide_mul_u128(a: u128, b: u128) -> (u128, u128) {
    #[inline]
    const fn to_low_high(x: u128) -> [u128; 2] {
        const MASK: u128 = u64::MAX as _;
        [x & MASK, x >> 64]
    }
    #[inline]
    const fn from_low_high(x: [u128; 2]) -> u128 {
        x[0] | (x[1] << 64)
    }
    #[inline]
    const fn scalar_mul(low_high: [u128; 2], k: u128) -> [u128; 3] {
        let [x, c] = to_low_high(k * low_high[0]);
        let [y, z] = to_low_high(k * low_high[1] + c);
        [x, y, z]
    }
    let a = to_low_high(a);
    let b = to_low_high(b);
    let low = scalar_mul(a, b[0]);
    let high = scalar_mul(a, b[1]);
    let r0 = low[0];
    let [r1, c] = to_low_high(low[1] + high[0]);
    let [r2, c] = to_low_high(low[2] + high[1] + c);
    let r3 = high[2] + c;
    (from_low_high([r0, r1]), from_low_high([r2, r3]))
}

#[rustc_const_unstable(feature = "core_intrinsics_fallbacks", issue = "none")]
impl const CarryingMulAdd for u128 {
    type Unsigned = u128;
    #[inline]
    fn carrying_mul_add(self, b: u128, c: u128, d: u128) -> (u128, u128) {
        let (low, mut high) = wide_mul_u128(self, b);
        let (low, carry) = u128::overflowing_add(low, c);
        high += carry as u128;
        let (low, carry) = u128::overflowing_add(low, d);
        high += carry as u128;
        (low, high)
    }
}

#[rustc_const_unstable(feature = "core_intrinsics_fallbacks", issue = "none")]
impl const CarryingMulAdd for i128 {
    type Unsigned = u128;
    #[inline]
    fn carrying_mul_add(self, b: i128, c: i128, d: i128) -> (u128, i128) {
        let (low, high) = wide_mul_u128(self as u128, b as u128);
        let mut high = high as i128;
        high = high.wrapping_add(i128::wrapping_mul(self >> 127, b));
        high = high.wrapping_add(i128::wrapping_mul(self, b >> 127));
        let (low, carry) = u128::overflowing_add(low, c as u128);
        high = high.wrapping_add((carry as i128) + (c >> 127));
        let (low, carry) = u128::overflowing_add(low, d as u128);
        high = high.wrapping_add((carry as i128) + (d >> 127));
        (low, high)
    }
}
