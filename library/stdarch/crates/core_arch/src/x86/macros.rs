//! Utility macros.
//!
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
        let _ = $crate::core_arch::x86::macros::ValidateConstRound::<$imm>::VALID;
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
        let _ = $crate::core_arch::x86::macros::ValidateConstSae::<$imm>::VALID;
    };
}

// Helper struct used to trigger const eval errors when the const generic immediate value `imm` is
// not a mantissas sae number.
pub(crate) struct ValidateConstMantissasSae<const IMM: i32>;
impl<const IMM: i32> ValidateConstMantissasSae<IMM> {
    pub(crate) const VALID: () = {
        assert!(IMM == 4 || IMM == 8 || IMM == 12, "Invalid IMM value");
    };
}

#[allow(unused)]
macro_rules! static_assert_mantissas_sae {
    ($imm:ident) => {
        let _ = $crate::core_arch::x86::macros::ValidateConstMantissasSae::<$imm>::VALID;
    };
}

// Helper struct used to trigger const eval errors when the unsigned const generic immediate value
// `IMM` is out of `[MIN-MAX]` range.
pub(crate) struct ValidateConstImmU32<const IMM: u32, const MIN: u32, const MAX: u32>;
impl<const IMM: u32, const MIN: u32, const MAX: u32> ValidateConstImmU32<IMM, MIN, MAX> {
    pub(crate) const VALID: () = {
        assert!(IMM >= MIN && IMM <= MAX, "IMM value not in expected range");
    };
}

#[allow(unused_macros)]
macro_rules! static_assert_imm_u8 {
    ($imm:ident) => {
        let _ =
            $crate::core_arch::x86::macros::ValidateConstImmU32::<$imm, 0, { (1 << 8) - 1 }>::VALID;
    };
}

// Helper struct used to trigger const eval errors when the const generic immediate value `SCALE` is
// not valid for gather instructions: the only valid scale values are 1, 2, 4 and 8.
pub(crate) struct ValidateConstGatherScale<const SCALE: i32>;
impl<const SCALE: i32> ValidateConstGatherScale<SCALE> {
    pub(crate) const VALID: () = {
        assert!(
            SCALE == 1 || SCALE == 2 || SCALE == 4 || SCALE == 8,
            "Invalid SCALE value"
        );
    };
}

#[allow(unused)]
macro_rules! static_assert_imm8_scale {
    ($imm:ident) => {
        let _ = $crate::core_arch::x86::macros::ValidateConstGatherScale::<$imm>::VALID;
    };
}

#[cfg(test)]
macro_rules! assert_approx_eq {
    ($a:expr, $b:expr, $eps:expr) => {{
        let (a, b) = (&$a, &$b);
        assert!(
            (*a - *b).abs() < $eps,
            "assertion failed: `(left !== right)` \
             (left: `{:?}`, right: `{:?}`, expect diff: `{:?}`, real diff: `{:?}`)",
            *a,
            *b,
            $eps,
            (*a - *b).abs()
        );
    }};
}
