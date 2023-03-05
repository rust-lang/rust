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

// Helper macro used to trigger const eval errors when the const generic immediate value `imm` is
// not a mantissas sae number.
#[allow(unused)]
macro_rules! static_assert_mantissas_sae {
    ($imm:ident) => {
        static_assert!($imm == 4 || $imm == 8 || $imm == 12, "Invalid IMM value")
    };
}

// Helper macro used to trigger const eval errors when the const generic immediate value `SCALE` is
// not valid for gather instructions: the only valid scale values are 1, 2, 4 and 8.
#[allow(unused)]
macro_rules! static_assert_imm8_scale {
    ($imm:ident) => {
        static_assert!(
            $imm == 1 || $imm == 2 || $imm == 4 || $imm == 8,
            "Invalid SCALE value"
        )
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
