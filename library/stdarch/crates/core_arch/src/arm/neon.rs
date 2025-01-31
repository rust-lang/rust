use crate::core_arch::arm_shared::neon::*;

#[allow(improper_ctypes)]
unsafe extern "unadjusted" {
    #[link_name = "llvm.arm.neon.vbsl.v8i8"]
    fn vbsl_s8_(a: int8x8_t, b: int8x8_t, c: int8x8_t) -> int8x8_t;
    #[link_name = "llvm.arm.neon.vbsl.v16i8"]
    fn vbslq_s8_(a: int8x16_t, b: int8x16_t, c: int8x16_t) -> int8x16_t;
}
