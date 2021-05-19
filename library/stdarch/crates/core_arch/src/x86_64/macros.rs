//! Utility macros.

// Helper struct used to trigger const eval errors when the const generic immediate value `imm` is
// not a round number.
pub(crate) struct ValidateConstRound<const IMM: i32>;
impl<const IMM: i32> ValidateConstRound<IMM> {
    pub(crate) const VALID: () = {
        assert!(
            IMM == 4 || IMM == 8 || IMM == 9 || IMM == 10 || IMM == 11,
            "Invalid IMM value"
        );
    };
}

#[allow(unused)]
macro_rules! static_assert_rounding {
    ($imm:ident) => {
        let _ = $crate::core_arch::x86_64::macros::ValidateConstRound::<$imm>::VALID;
    };
}

// Helper struct used to trigger const eval errors when the const generic immediate value `imm` is
// not a sae number.
pub(crate) struct ValidateConstSae<const IMM: i32>;
impl<const IMM: i32> ValidateConstSae<IMM> {
    pub(crate) const VALID: () = {
        assert!(IMM == 4 || IMM == 8, "Invalid IMM value");
    };
}

#[allow(unused)]
macro_rules! static_assert_sae {
    ($imm:ident) => {
        let _ = $crate::core_arch::x86_64::macros::ValidateConstSae::<$imm>::VALID;
    };
}
