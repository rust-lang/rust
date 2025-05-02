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
// not an extended rounding number
#[allow(unused)]
macro_rules! static_assert_extended_rounding {
    ($imm: ident) => {
        static_assert!(($imm & 7) < 5 && ($imm & !15) == 0, "Invalid IMM value")
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

// x86-32 wants to use a 32-bit address size, but asm! defaults to using the full
// register name (e.g. rax). We have to explicitly override the placeholder to
// use the 32-bit register name in that case.

#[cfg(target_pointer_width = "32")]
macro_rules! vpl {
    ($inst:expr) => {
        concat!($inst, ", [{p:e}]")
    };
}
#[cfg(target_pointer_width = "64")]
macro_rules! vpl {
    ($inst:expr) => {
        concat!($inst, ", [{p}]")
    };
}

#[cfg(target_pointer_width = "32")]
macro_rules! vps {
    ($inst1:expr, $inst2:expr) => {
        concat!($inst1, " [{p:e}]", $inst2)
    };
}
#[cfg(target_pointer_width = "64")]
macro_rules! vps {
    ($inst1:expr, $inst2:expr) => {
        concat!($inst1, " [{p}]", $inst2)
    };
}
