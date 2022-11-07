// Source: musl libm rintf
// (equivalent to roundevenf when rounding mode is default,
// which Rust assumes)

#[cfg_attr(all(test, assert_no_panic), no_panic::no_panic)]
pub fn roundevenf(x: f32) -> f32 {
    let one_over_e = 1.0 / f32::EPSILON;
    let as_u32: u32 = x.to_bits();
    let exponent: u32 = as_u32 >> 23 & 0xff;
    let is_positive = (as_u32 >> 31) == 0;
    if exponent >= 0x7f + 23 {
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
    use super::roundevenf;

    #[test]
    fn negative_zero() {
        assert_eq!(roundevenf(-0.0_f32).to_bits(), (-0.0_f32).to_bits());
    }

    #[test]
    fn sanity_check() {
        assert_eq!(roundevenf(-1.0), -1.0);
        assert_eq!(roundevenf(2.8), 3.0);
        assert_eq!(roundevenf(-0.5), -0.0);
        assert_eq!(roundevenf(0.5), 0.0);
        assert_eq!(roundevenf(-1.5), -2.0);
        assert_eq!(roundevenf(1.5), 2.0);
    }
}
