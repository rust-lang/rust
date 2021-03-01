//! Utility macros.

macro_rules! static_assert_imm_s5 {
    ($imm:ident) => {
        let _ = $crate::core_arch::macros::ValidateConstImm::<$imm, -16, 15>::VALID;
    };
}

macro_rules! static_assert_imm_s10 {
    ($imm:ident) => {
        let _ = $crate::core_arch::macros::ValidateConstImm::<$imm, -512, 511>::VALID;
    };
}

macro_rules! static_assert_imm_s11 {
    ($imm:ident) => {
        let _ = $crate::core_arch::macros::ValidateConstImm::<$imm, -1024, 1023>::VALID;
    };
}

macro_rules! static_assert_imm_s12 {
    ($imm:ident) => {
        let _ = $crate::core_arch::macros::ValidateConstImm::<$imm, -2048, 2047>::VALID;
    };
}

macro_rules! static_assert_imm_s13 {
    ($imm:ident) => {
        let _ = $crate::core_arch::macros::ValidateConstImm::<$imm, -4096, 4095>::VALID;
    };
}
