use super::exp;

const K: u32 = 2043;
const KLN2: f64 = 1416.0996898839683;

#[inline]
pub(crate) fn k_expo2(x: f64) -> f64 {
    /* note that k is odd and scale*scale overflows */
    let scale = f64::from_bits((((0x3ff + K / 2) << 20) as u64) << 32);
    /* exp(x - k ln2) * 2**(k-1) */
    return exp(x - KLN2) * scale * scale;
}
