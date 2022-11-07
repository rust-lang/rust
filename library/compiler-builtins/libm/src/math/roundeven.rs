// Source: musl libm rint
// (equivalent to roundeven when rounding mode is default,
// which Rust assumes)

#[cfg_attr(all(test, assert_no_panic), no_panic::no_panic)]
pub fn roundeven(x: f64) -> f64 {
    let one_over_e = 1.0 / f64::EPSILON;
    let as_u64: u64 = x.to_bits();
    let exponent: u64 = as_u64 >> 52 & 0x7ff;
    let is_positive = (as_u64 >> 63) == 0;
    if exponent >= 0x3ff + 52 {
        x
    } else {
        let ans = if is_positive {
            x + one_over_e - one_over_e
        } else {
            x - one_over_e + one_over_e
        };

        if ans == 0.0 {
            if is_positive {
                0.0
            } else {
                -0.0
            }
        } else {
            ans
        }
    }
}

#[cfg(test)]
mod tests {
    use super::roundeven;

    #[test]
    fn negative_zero() {
        assert_eq!(roundeven(-0.0_f64).to_bits(), (-0.0_f64).to_bits());
    }

    #[test]
    fn sanity_check() {
        assert_eq!(roundeven(-1.0), -1.0);
        assert_eq!(roundeven(2.8), 3.0);
        assert_eq!(roundeven(-0.5), -0.0);
        assert_eq!(roundeven(0.5), 0.0);
        assert_eq!(roundeven(-1.5), -2.0);
        assert_eq!(roundeven(1.5), 2.0);
    }
}
