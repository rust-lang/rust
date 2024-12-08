// We're testing x86 target specific features
//@only-target: x86_64 i686
//@compile-flags: -C target-feature=+adx

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    #[cfg(target_arch = "x86")]
    use core::arch::x86 as arch;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64 as arch;

    fn adc(c_in: u8, a: u32, b: u32) -> (u8, u32) {
        let mut sum = 0;
        // SAFETY: There are no safety requirements for calling `_addcarry_u32`.
        // It's just unsafe for API consistency with other intrinsics.
        let c_out = unsafe { arch::_addcarryx_u32(c_in, a, b, &mut sum) };
        (c_out, sum)
    }

    pub fn main() {
        assert_eq!(adc(0, 1, 1), (0, 2));
        assert_eq!(adc(1, 1, 1), (0, 3));
        assert_eq!(adc(2, 1, 1), (0, 3)); // any non-zero carry acts as 1!
        assert_eq!(adc(u8::MAX, 1, 1), (0, 3));
        assert_eq!(adc(0, u32::MAX, u32::MAX), (1, u32::MAX - 1));
        assert_eq!(adc(1, u32::MAX, u32::MAX), (1, u32::MAX));
        assert_eq!(adc(2, u32::MAX, u32::MAX), (1, u32::MAX));
        assert_eq!(adc(u8::MAX, u32::MAX, u32::MAX), (1, u32::MAX));
    }
}

#[cfg(target_arch = "x86_64")]
mod x86_64 {
    use core::arch::x86_64 as arch;

    fn adc(c_in: u8, a: u64, b: u64) -> (u8, u64) {
        let mut sum = 0;
        // SAFETY: There are no safety requirements for calling `_addcarry_u64`.
        // It's just unsafe for API consistency with other intrinsics.
        let c_out = unsafe { arch::_addcarryx_u64(c_in, a, b, &mut sum) };
        (c_out, sum)
    }

    pub fn main() {
        assert_eq!(adc(0, 1, 1), (0, 2));
        assert_eq!(adc(1, 1, 1), (0, 3));
        assert_eq!(adc(2, 1, 1), (0, 3)); // any non-zero carry acts as 1!
        assert_eq!(adc(u8::MAX, 1, 1), (0, 3));
        assert_eq!(adc(0, u64::MAX, u64::MAX), (1, u64::MAX - 1));
        assert_eq!(adc(1, u64::MAX, u64::MAX), (1, u64::MAX));
        assert_eq!(adc(2, u64::MAX, u64::MAX), (1, u64::MAX));
        assert_eq!(adc(u8::MAX, u64::MAX, u64::MAX), (1, u64::MAX));
    }
}

fn main() {
    assert!(is_x86_feature_detected!("adx"));

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    x86::main();
    #[cfg(target_arch = "x86_64")]
    x86_64::main();
}
