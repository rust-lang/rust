use core::f64;

const TOINT: f64 = 1. / f64::EPSILON;

/// Ceil (f64)
///
/// Finds the nearest integer greater than or equal to `x`.
#[inline]
#[cfg_attr(all(test, assert_no_panic), no_panic::no_panic)]
pub fn ceil(x: f64) -> f64 {
    // On wasm32 we know that LLVM's intrinsic will compile to an optimized
    // `f64.ceil` native instruction, so we can leverage this for both code size
    // and speed.
    llvm_intrinsically_optimized! {
        #[cfg(target_arch = "wasm32")] {
            return unsafe { ::core::intrinsics::ceilf64(x) }
        }
    }
    let u: u64 = x.to_bits();
    let e: i64 = (u >> 52 & 0x7ff) as i64;
    let y: f64;

    if e >= 0x3ff + 52 || x == 0. {
        return x;
    }
    // y = int(x) - x, where int(x) is an integer neighbor of x
    y = if (u >> 63) != 0 {
        x - TOINT + TOINT - x
    } else {
        x + TOINT - TOINT - x
    };
    // special case because of non-nearest rounding modes
    if e < 0x3ff {
        force_eval!(y);
        return if (u >> 63) != 0 { -0. } else { 1. };
    }
    if y < 0. {
        x + y + 1.
    } else {
        x + y
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn sanity_check() {
        assert_eq!(super::ceil(1.1), 2.0);
        assert_eq!(super::ceil(2.9), 3.0);
    }
}
