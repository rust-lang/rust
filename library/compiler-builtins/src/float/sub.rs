use float::add::__adddf3;
use float::add::__addsf3;
use float::Float;

intrinsics! {
    #[arm_aeabi_alias = __aeabi_fsub]
    pub extern "C" fn __subsf3(a: f32, b: f32) -> f32 {
        __addsf3(a, f32::from_repr(b.repr() ^ f32::SIGN_MASK))
    }

    #[arm_aeabi_alias = __aeabi_dsub]
    pub extern "C" fn __subdf3(a: f64, b: f64) -> f64 {
        __adddf3(a, f64::from_repr(b.repr() ^ f64::SIGN_MASK))
    }

    #[cfg(target_arch = "arm")]
    pub extern "C" fn __subsf3vfp(a: f32, b: f32) -> f32 {
        a - b
    }

    #[cfg(target_arch = "arm")]
    pub extern "C" fn __subdf3vfp(a: f64, b: f64) -> f64 {
        a - b
    }
}
