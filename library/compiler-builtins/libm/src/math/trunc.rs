use core::f64;

/// Rounds the number toward 0 to the closest integral value (f64).
///
/// This effectively removes the decimal part of the number, leaving the integral part.
#[cfg_attr(all(test, assert_no_panic), no_panic::no_panic)]
pub fn trunc(x: f64) -> f64 {
    select_implementation! {
        name: trunc,
        use_arch: all(target_arch = "wasm32", intrinsics_enabled),
        args: x,
    }

    let x1p120 = f64::from_bits(0x4770000000000000); // 0x1p120f === 2 ^ 120

    let mut i: u64 = x.to_bits();
    let mut e: i64 = ((i >> 52) & 0x7ff) as i64 - 0x3ff + 12;
    let m: u64;

    if e >= 52 + 12 {
        return x;
    }
    if e < 12 {
        e = 1;
    }
    m = -1i64 as u64 >> e;
    if (i & m) == 0 {
        return x;
    }
    force_eval!(x + x1p120);
    i &= !m;
    f64::from_bits(i)
}

#[cfg(test)]
mod tests {
    #[test]
    fn sanity_check() {
        assert_eq!(super::trunc(1.1), 1.0);
    }
}
