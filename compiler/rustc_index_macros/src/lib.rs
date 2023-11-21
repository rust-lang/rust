#![cfg_attr(feature = "nightly", feature(allow_internal_unstable))]
#![cfg_attr(feature = "nightly", allow(internal_features))]

use proc_macro::TokenStream;

mod newtype;

/// Creates a struct type `S` that can be used as an index with
/// `IndexVec` and so on.
///
/// There are two ways of interacting with these indices:
///
/// - The `From` impls are the preferred way. So you can do
///   `S::from(v)` with a `usize` or `u32`. And you can convert back
///   to an integer with `u32::from(s)`.
///
/// - Alternatively, you can use the methods `S::new(v)` and `s.index()`
///   to create/return a value.
///
/// Internally, the index uses a u32, so the index must not exceed
/// `u32::MAX`. You can also customize things like the `Debug` impl,
/// what traits are derived, and so forth via the macro.
#[proc_macro]
#[cfg_attr(
    feature = "nightly",
    allow_internal_unstable(step_trait, rustc_attrs, trusted_step, spec_option_partial_eq)
)]
pub fn newtype_index(input: TokenStream) -> TokenStream {
    newtype::newtype(input)
}
