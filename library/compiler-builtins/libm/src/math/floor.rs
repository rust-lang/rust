use core::f64;

const TOINT    : f64 = 1. / f64::EPSILON;

#[inline]
pub fn floor(x : f64) -> f64 {
    let ui = x.to_bits();
	let e = ((ui >> 52) & 0x7ff) as i32;

	if (e >= 0x3ff+52) || (x == 0.) {
		return x;
    }
	/* y = int(x) - x, where int(x) is an integer neighbor of x */
	let y = if (ui >> 63) != 0 {
		x - TOINT + TOINT - x
	} else {
		x + TOINT - TOINT - x
    };
	/* special case because of non-nearest rounding modes */
	if e <= 0x3ff-1 {
		force_eval!(y);
		return if (ui >> 63) != 0 { -1. } else { 0. };
	}
	if y > 0. {
        x + y - 1.
    } else {
        x + y
    }
}
