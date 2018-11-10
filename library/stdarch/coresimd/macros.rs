//! Utility macros.

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
        #[cfg_attr(feature = "cargo-clippy",
                   allow(clippy::missing_inline_in_public_items))]
        pub struct $name($($fields)*);
    )*)
}
