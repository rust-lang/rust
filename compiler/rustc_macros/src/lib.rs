// tidy-alphabetical-start
#![allow(rustc::default_hash_types)]
#![feature(if_let_guard)]
#![feature(never_type)]
#![feature(proc_macro_diagnostic)]
#![feature(proc_macro_span)]
#![feature(proc_macro_tracked_env)]
// tidy-alphabetical-end

use proc_macro::TokenStream;
use synstructure::decl_derive;

mod current_version;
mod diagnostics;
mod extension;
mod hash_stable;
mod lift;
mod print_attribute;
mod query;
mod serialize;
mod symbols;
mod try_from;
mod type_foldable;
mod type_visitable;

// Reads the rust version (e.g. "1.75.0") from the CFG_RELEASE env var and
// produces a `RustcVersion` literal containing that version (e.g.
// `RustcVersion { major: 1, minor: 75, patch: 0 }`).
#[proc_macro]
pub fn current_rustc_version(input: TokenStream) -> TokenStream {
    current_version::current_version(input)
}

#[proc_macro]
pub fn rustc_queries(input: TokenStream) -> TokenStream {
    query::rustc_queries(input)
}

#[proc_macro]
pub fn symbols(input: TokenStream) -> TokenStream {
    symbols::symbols(input.into()).into()
}

/// Derive an extension trait for a given impl block. The trait name
/// goes into the parenthesized args of the macro, for greppability.
/// For example:
/// ```
/// use rustc_macros::extension;
/// #[extension(pub trait Foo)]
/// impl i32 { fn hello() {} }
/// ```
///
/// expands to:
/// ```
/// pub trait Foo { fn hello(); }
/// impl Foo for i32 { fn hello() {} }
/// ```
#[proc_macro_attribute]
pub fn extension(attr: TokenStream, input: TokenStream) -> TokenStream {
    extension::extension(attr, input)
}

decl_derive!([HashStable, attributes(stable_hasher)] => hash_stable::hash_stable_derive);
decl_derive!(
    [HashStable_Generic, attributes(stable_hasher)] =>
    hash_stable::hash_stable_generic_derive
);
decl_derive!(
    [HashStable_NoContext] =>
    /// `HashStable` implementation that has no `HashStableContext` bound and
    /// which adds `where` bounds for `HashStable` based off of fields and not
    /// generics. This is suitable for use in crates like `rustc_type_ir`.
    hash_stable::hash_stable_no_context_derive
);

decl_derive!([Decodable_NoContext] => serialize::decodable_nocontext_derive);
decl_derive!([Encodable_NoContext] => serialize::encodable_nocontext_derive);
decl_derive!([Decodable] => serialize::decodable_derive);
decl_derive!([Encodable] => serialize::encodable_derive);
decl_derive!([TyDecodable] => serialize::type_decodable_derive);
decl_derive!([TyEncodable] => serialize::type_encodable_derive);
decl_derive!([MetadataDecodable] => serialize::meta_decodable_derive);
decl_derive!([MetadataEncodable] => serialize::meta_encodable_derive);
decl_derive!(
    [TypeFoldable, attributes(type_foldable)] =>
    /// Derives `TypeFoldable` for the annotated `struct` or `enum` (`union` is not supported).
    ///
    /// The fold will produce a value of the same struct or enum variant as the input, with
    /// each field respectively folded using the `TypeFoldable` implementation for its type.
    /// However, if a field of a struct or an enum variant is annotated with
    /// `#[type_foldable(identity)]` then that field will retain its incumbent value (and its
    /// type is not required to implement `TypeFoldable`).
    type_foldable::type_foldable_derive
);
decl_derive!(
    [TypeVisitable, attributes(type_visitable)] =>
    /// Derives `TypeVisitable` for the annotated `struct` or `enum` (`union` is not supported).
    ///
    /// Each field of the struct or enum variant will be visited in definition order, using the
    /// `TypeVisitable` implementation for its type. However, if a field of a struct or an enum
    /// variant is annotated with `#[type_visitable(ignore)]` then that field will not be
    /// visited (and its type is not required to implement `TypeVisitable`).
    type_visitable::type_visitable_derive
);
decl_derive!([Lift, attributes(lift)] => lift::lift_derive);
decl_derive!(
    [Diagnostic, attributes(
        // struct attributes
        diag,
        help,
        help_once,
        note,
        note_once,
        warning,
        // field attributes
        skip_arg,
        primary_span,
        label,
        subdiagnostic,
        suggestion,
        suggestion_short,
        suggestion_hidden,
        suggestion_verbose)] => diagnostics::diagnostic_derive
);
decl_derive!(
    [LintDiagnostic, attributes(
        // struct attributes
        diag,
        help,
        help_once,
        note,
        note_once,
        warning,
        // field attributes
        skip_arg,
        primary_span,
        label,
        subdiagnostic,
        suggestion,
        suggestion_short,
        suggestion_hidden,
        suggestion_verbose)] => diagnostics::lint_diagnostic_derive
);
decl_derive!(
    [Subdiagnostic, attributes(
        // struct/variant attributes
        label,
        help,
        help_once,
        note,
        note_once,
        warning,
        subdiagnostic,
        suggestion,
        suggestion_short,
        suggestion_hidden,
        suggestion_verbose,
        multipart_suggestion,
        multipart_suggestion_short,
        multipart_suggestion_hidden,
        multipart_suggestion_verbose,
        // field attributes
        skip_arg,
        primary_span,
        suggestion_part,
        applicability)] => diagnostics::subdiagnostic_derive
);

decl_derive! {
    [TryFromU32] =>
    /// Derives `TryFrom<u32>` for the annotated `enum`, which must have no fields.
    /// Each variant maps to the value it would produce under an `as u32` cast.
    ///
    /// The error type is `u32`.
    try_from::try_from_u32
}
decl_derive! {
    [PrintAttribute] =>
    /// Derives `PrintAttribute` for `AttributeKind`.
    /// This macro is pretty specific to `rustc_attr_data_structures` and likely not that useful in
    /// other places. It's deriving something close to `Debug` without printing some extraenous
    /// things like spans.
    print_attribute::print_attribute
}
