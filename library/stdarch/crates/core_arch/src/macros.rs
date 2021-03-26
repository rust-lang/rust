//! Utility macros.

// Helper struct used to trigger const eval errors when the const generic immediate value `IMM` is
// out of `[MIN-MAX]` range.
pub(crate) struct ValidateConstImm<const IMM: i32, const MIN: i32, const MAX: i32>;
impl<const IMM: i32, const MIN: i32, const MAX: i32> ValidateConstImm<IMM, MIN, MAX> {
    pub(crate) const VALID: () = {
        let _ = 1 / ((IMM >= MIN && IMM <= MAX) as usize);
    };
}

#[allow(unused_macros)]
macro_rules! static_assert_imm1 {
    ($imm:ident) => {
        let _ = $crate::core_arch::macros::ValidateConstImm::<$imm, 0, { (1 << 1) - 1 }>::VALID;
    };
}

#[allow(unused_macros)]
macro_rules! static_assert_imm2 {
    ($imm:ident) => {
        let _ = $crate::core_arch::macros::ValidateConstImm::<$imm, 0, { (1 << 2) - 1 }>::VALID;
    };
}

#[allow(unused_macros)]
macro_rules! static_assert_imm3 {
    ($imm:ident) => {
        let _ = $crate::core_arch::macros::ValidateConstImm::<$imm, 0, { (1 << 3) - 1 }>::VALID;
    };
}

#[allow(unused_macros)]
macro_rules! static_assert_imm4 {
    ($imm:ident) => {
        let _ = $crate::core_arch::macros::ValidateConstImm::<$imm, 0, { (1 << 4) - 1 }>::VALID;
    };
}

#[allow(unused_macros)]
macro_rules! static_assert_imm5 {
    ($imm:ident) => {
        let _ = $crate::core_arch::macros::ValidateConstImm::<$imm, 0, { (1 << 5) - 1 }>::VALID;
    };
}

#[allow(unused_macros)]
macro_rules! static_assert_imm6 {
    ($imm:ident) => {
        let _ = $crate::core_arch::macros::ValidateConstImm::<$imm, 0, { (1 << 6) - 1 }>::VALID;
    };
}

#[allow(unused_macros)]
macro_rules! static_assert_imm8 {
    ($imm:ident) => {
        let _ = $crate::core_arch::macros::ValidateConstImm::<$imm, 0, { (1 << 8) - 1 }>::VALID;
    };
}

#[allow(unused_macros)]
macro_rules! static_assert_imm16 {
    ($imm:ident) => {
        let _ = $crate::core_arch::macros::ValidateConstImm::<$imm, 0, { (1 << 16) - 1 }>::VALID;
    };
}

#[allow(unused)]
macro_rules! static_assert {
    ($imm:ident : $ty:ty where $e:expr) => {{
        struct Validate<const $imm: $ty>();
        impl<const $imm: $ty> Validate<$imm> {
            const VALID: () = {
                let _ = 1 / ($e as usize);
            };
        }
        let _ = Validate::<$imm>::VALID;
    }};
}

#[allow(unused)]
macro_rules! types {
    ($(
        $(#[$doc:meta])*
        pub struct $name:ident($($fields:tt)*);
    )*) => ($(
        $(#[$doc])*
        #[derive(Copy, Clone, Debug)]
        #[allow(non_camel_case_types)]
        #[repr(simd)]
        #[allow(clippy::missing_inline_in_public_items)]
        pub struct $name($($fields)*);
    )*)
}
