use core::f32;

const TOINT: f32 = 1.0 / f32::EPSILON;

#[inline]
pub fn roundf(mut x: f32) -> f32 {
    let i = x.to_bits();
    let e: u32 = i >> 23 & 0xff;
    let mut y: f32;

    if e >= 0x7f + 23 {
        return x;
    }
    if i >> 31 != 0 {
        x = -x;
    }
    if e < 0x7f - 1 {
        force_eval!(x + TOINT);
        return 0.0 * x;
    }
    y = x + TOINT - TOINT - x;
    if y > 0.5f32 {
        y = y + x - 1.0;
    } else if y <= -0.5f32 {
        y = y + x + 1.0;
    } else {
        y = y + x;
    }
    if i >> 31 != 0 {
        -y
    } else {
        y
    }
}
