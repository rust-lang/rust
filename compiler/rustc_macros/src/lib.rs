#![feature(allow_internal_unstable)]
#![feature(let_else)]
#![feature(proc_macro_diagnostic)]
#![allow(rustc::default_hash_types)]
#![recursion_limit = "128"]

use synstructure::decl_derive;

use proc_macro::TokenStream;

mod diagnostics;
mod hash_stable;
mod lift;
mod newtype;
mod query;
mod serialize;
mod symbols;
mod type_foldable;

#[proc_macro]
pub fn rustc_queries(input: TokenStream) -> TokenStream {
    query::rustc_queries(input)
}

#[proc_macro]
pub fn symbols(input: TokenStream) -> TokenStream {
    symbols::symbols(input.into()).into()
}

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
#[allow_internal_unstable(step_trait, rustc_attrs, trusted_step)]
pub fn newtype_index(input: TokenStream) -> TokenStream {
    newtype::newtype(input)
}

decl_derive!([HashStable, attributes(stable_hasher)] => hash_stable::hash_stable_derive);
decl_derive!(
    [HashStable_Generic, attributes(stable_hasher)] =>
    hash_stable::hash_stable_generic_derive
);

decl_derive!([Decodable] => serialize::decodable_derive);
decl_derive!([Encodable] => serialize::encodable_derive);
decl_derive!([TyDecodable] => serialize::type_decodable_derive);
decl_derive!([TyEncodable] => serialize::type_encodable_derive);
decl_derive!([MetadataDecodable] => serialize::meta_decodable_derive);
decl_derive!([MetadataEncodable] => serialize::meta_encodable_derive);
decl_derive!([TypeFoldable, attributes(type_foldable)] => type_foldable::type_foldable_derive);
decl_derive!([Lift, attributes(lift)] => lift::lift_derive);
decl_derive!(
    [SessionDiagnostic, attributes(
        // struct attributes
        warning,
        error,
        note,
        help,
        // field attributes
        skip_arg,
        primary_span,
        label,
        subdiagnostic,
        suggestion,
        suggestion_short,
        suggestion_hidden,
        suggestion_verbose)] => diagnostics::session_diagnostic_derive
);
decl_derive!(
    [SessionSubdiagnostic, attributes(
        // struct/variant attributes
        label,
        help,
        note,
        suggestion,
        suggestion_short,
        suggestion_hidden,
        suggestion_verbose,
        // field attributes
        skip_arg,
        primary_span,
        applicability)] => diagnostics::session_subdiagnostic_derive
);
