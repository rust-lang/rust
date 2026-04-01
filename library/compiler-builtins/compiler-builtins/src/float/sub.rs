use crate::float::Float;

intrinsics! {
    #[cfg(f16_enabled)]
    pub extern "C" fn __subhf3(a: f16, b: f16) -> f16 {
        crate::float::add::__addhf3(a, f16::from_bits(b.to_bits() ^ f16::SIGN_MASK))
    }

    #[arm_aeabi_alias = __aeabi_fsub]
    pub extern "C" fn __subsf3(a: f32, b: f32) -> f32 {
        crate::float::add::__addsf3(a, f32::from_bits(b.to_bits() ^ f32::SIGN_MASK))
    }

    #[arm_aeabi_alias = __aeabi_dsub]
    pub extern "C" fn __subdf3(a: f64, b: f64) -> f64 {
        crate::float::add::__adddf3(a, f64::from_bits(b.to_bits() ^ f64::SIGN_MASK))
    }

    #[ppc_alias = __subkf3]
    #[cfg(f128_enabled)]
    pub extern "C" fn __subtf3(a: f128, b: f128) -> f128 {
        #[cfg(any(target_arch = "powerpc", target_arch = "powerpc64"))]
        use crate::float::add::__addkf3 as __addtf3;
        #[cfg(not(any(target_arch = "powerpc", target_arch = "powerpc64")))]
        use crate::float::add::__addtf3;

        __addtf3(a, f128::from_bits(b.to_bits() ^ f128::SIGN_MASK))
    }
}
