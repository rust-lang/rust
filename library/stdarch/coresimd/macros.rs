//! Utility macros.

#[allow(unused)]
macro_rules! types {
    ($(
        $(#[$doc:meta])*
        pub struct $name:ident($($fields:tt)*);
    )*) => ($(
        $(#[$doc])*
            #[derive(Copy, Debug)]
        #[allow(non_camel_case_types)]
        #[repr(simd)]
        pub struct $name($($fields)*);

        #[cfg_attr(feature = "cargo-clippy", allow(expl_impl_clone_on_copy))]
        impl ::clone::Clone for $name {
            #[inline] // currently needed for correctness
            fn clone(&self) -> $name {
                *self
            }
        }
    )*)
}
