use core::u64;

#[inline]
pub fn fabs(x: f64) -> f64 {
    f64::from_bits(x.to_bits() & (u64::MAX / 2))
}
