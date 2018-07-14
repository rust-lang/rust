use core::f32;

#[inline]
pub fn ceilf(x: f32) -> f32 {
    let mut ui = x.to_bits();
    let e = (((ui >> 23) & 0xff) - 0x7f) as i32;

    if e >= 23 {
        return x;
    }
    if e >= 0 {
        let m = 0x007fffff >> e;
        if (ui & m) == 0 {
            return x;
        }
        force_eval!(x + f32::from_bits(0x7b800000));
        if ui >> 31 == 0 {
            ui += m;
        }
        ui &= !m;
    } else {
        force_eval!(x + f32::from_bits(0x7b800000));
        if ui >> 31 != 0 {
            return -0.0;
        } else if ui << 1 != 0 {
            return 1.0;
        }
    }
    return f32::from_bits(ui);
}
