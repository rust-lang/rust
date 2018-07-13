#[inline]
pub fn scalbnf(mut x: f32, mut n : i32) -> f32 {
    let x1p127  = f32::from_bits(0x7f000000); // 0x1p127f === 2 ^ 127
    let x1p_126 = f32::from_bits(0x800000);   // 0x1p-126f === 2 ^ -126
    let x1p24   = f32::from_bits(0x4b800000); // 0x1p24f === 2 ^ 24
    
    if n > 127 {
        x *= x1p127;
        n -= 127;
        if n > 127 {
            x *= x1p127;
            n -= 127;
            if n > 127 {
                n = 127;
            }
        }
    } else if n < -126 {
        x *= x1p_126 * x1p24;
        n += 126 - 24;
        if n < -126 {
            x *= x1p_126 * x1p24;
            n += 126 - 24;
            if n < -126 {
                n = -126;
            }
        }
    }
    x * f32::from_bits(((0x7f+n) as u32)<<23)
}
