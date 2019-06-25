use core::f32;

/// Floor (f32)
///
/// Finds the nearest integer less than or equal to `x`.
#[inline]
#[cfg_attr(all(test, assert_no_panic), no_panic::no_panic)]
pub fn floorf(x: f32) -> f32 {
    // On wasm32 we know that LLVM's intrinsic will compile to an optimized
    // `f32.floor` native instruction, so we can leverage this for both code size
    // and speed.
    llvm_intrinsically_optimized! {
        #[cfg(target_arch = "wasm32")] {
            return unsafe { ::core::intrinsics::floorf32(x) }
        }
    }
    let mut ui = x.to_bits();
    let e = (((ui >> 23) as i32) & 0xff) - 0x7f;

    if e >= 23 {
        return x;
    }
    if e >= 0 {
        let m: u32 = 0x007fffff >> e;
        if (ui & m) == 0 {
            return x;
        }
        force_eval!(x + f32::from_bits(0x7b800000));
        if ui >> 31 != 0 {
            ui += m;
        }
        ui &= !m;
    } else {
        force_eval!(x + f32::from_bits(0x7b800000));
        if ui >> 31 == 0 {
            ui = 0;
        } else if ui << 1 != 0 {
            return -1.0;
        }
    }
    f32::from_bits(ui)
}

#[cfg(test)]
mod tests {
    #[test]
    fn no_overflow() {
        assert_eq!(super::floorf(0.5), 0.0);
    }
}
