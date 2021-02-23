//! Utility macros.

// For round instructions, the only valid values for rounding are 4, 8, 9, 10 and 11.
// This macro enforces that.
#[allow(unused)]
macro_rules! constify_imm4_round {
    ($imm8:expr, $expand:ident) => {
        #[allow(overflowing_literals)]
        match ($imm8) & 0b1111 {
            4 => $expand!(4),
            8 => $expand!(8),
            9 => $expand!(9),
            10 => $expand!(10),
            11 => $expand!(11),
            _ => panic!("Invalid round value"),
        }
    };
}

// For sae instructions, the only valid values for sae are 4 and 8.
// This macro enforces that.
#[allow(unused)]
macro_rules! constify_imm4_sae {
    ($imm8:expr, $expand:ident) => {
        #[allow(overflowing_literals)]
        match ($imm8) & 0b1111 {
            4 => $expand!(4),
            8 => $expand!(8),
            _ => panic!("Invalid sae value"),
        }
    };
}
