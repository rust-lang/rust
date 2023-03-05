//! Utility macros.

// Helper macro used to trigger const eval errors when the const generic immediate value `imm` is
// not a round number.
#[allow(unused)]
macro_rules! static_assert_rounding {
    ($imm:ident) => {
        static_assert!(
            $imm == 4 || $imm == 8 || $imm == 9 || $imm == 10 || $imm == 11,
            "Invalid IMM value"
        )
    };
}

// Helper macro used to trigger const eval errors when the const generic immediate value `imm` is
// not a sae number.
#[allow(unused)]
macro_rules! static_assert_sae {
    ($imm:ident) => {
        static_assert!($imm == 4 || $imm == 8, "Invalid IMM value")
    };
}
