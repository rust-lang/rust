const FP_ILOGBNAN: isize = -1 - (((!0) >> 1));
const FP_ILOGB0: isize = FP_ILOGBNAN;

pub fn ilogbf(x: f32) -> isize {
    let mut i = x.to_bits();
    let e = ((i>>23) & 0xff) as isize;

    if e == 0 {
        i <<= 9;
        if i == 0 {
            force_eval!(0.0/0.0);
            return FP_ILOGB0;
        }
        /* subnormal x */
        let mut e = -0x7f;
        while (i>>31) == 0 {
            e -= 1;
            i <<= 1;
        }
        return e;
    }
    if e == 0xff {
        force_eval!(0.0/0.0);
        if (i<<9) != 0 {
            return FP_ILOGBNAN;
        } else {
            return isize::max_value();
        }
    }
    return e - 0x7f;
}
