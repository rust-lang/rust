#[inline]
pub fn fabsf(x: f32) -> f32 {
    f32::from_bits(x.to_bits() & 0x7fffffff)
}
