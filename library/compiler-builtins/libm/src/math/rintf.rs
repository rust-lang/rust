#[cfg_attr(all(test, assert_no_panic), no_panic::no_panic)]
pub fn rintf(x: f32) -> f32 {
    select_implementation! {
        name: rintf,
        use_arch: all(target_arch = "wasm32", intrinsics_enabled),
        args: x,
    }

    let one_over_e = 1.0 / f32::EPSILON;
    let as_u32: u32 = x.to_bits();
    let exponent: u32 = (as_u32 >> 23) & 0xff;
    let is_positive = (as_u32 >> 31) == 0;
    if exponent >= 0x7f + 23 {
        x
    } else {
        let ans = if is_positive {
            #[cfg(all(target_arch = "x86", not(target_feature = "sse2")))]
            let x = force_eval!(x);
            let xplusoneovere = x + one_over_e;
            #[cfg(all(target_arch = "x86", not(target_feature = "sse2")))]
            let xplusoneovere = force_eval!(xplusoneovere);
            xplusoneovere - one_over_e
        } else {
            #[cfg(all(target_arch = "x86", not(target_feature = "sse2")))]
            let x = force_eval!(x);
            let xminusoneovere = x - one_over_e;
            #[cfg(all(target_arch = "x86", not(target_feature = "sse2")))]
            let xminusoneovere = force_eval!(xminusoneovere);
            xminusoneovere + one_over_e
        };

        if ans == 0.0 { if is_positive { 0.0 } else { -0.0 } } else { ans }
    }
}

// PowerPC tests are failing on LLVM 13: https://github.com/rust-lang/rust/issues/88520
#[cfg(not(target_arch = "powerpc64"))]
#[cfg(test)]
mod tests {
    use super::rintf;

    #[test]
    fn negative_zero() {
        assert_eq!(rintf(-0.0_f32).to_bits(), (-0.0_f32).to_bits());
    }

    #[test]
    fn sanity_check() {
        assert_eq!(rintf(-1.0), -1.0);
        assert_eq!(rintf(2.8), 3.0);
        assert_eq!(rintf(-0.5), -0.0);
        assert_eq!(rintf(0.5), 0.0);
        assert_eq!(rintf(-1.5), -2.0);
        assert_eq!(rintf(1.5), 2.0);
    }
}
