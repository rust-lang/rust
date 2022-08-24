#![feature(allow_internal_unstable)]
#![feature(let_else)]
#![feature(never_type)]
#![feature(proc_macro_diagnostic)]
#![feature(proc_macro_span)]
#![allow(rustc::default_hash_types)]
#![deny(rustc::untranslatable_diagnostic)]
#![deny(rustc::diagnostic_outside_of_impl)]
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
mod type_visitable;

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

/// Implements the `fluent_messages` macro, which performs compile-time validation of the
/// compiler's Fluent resources (i.e. that the resources parse and don't multiply define the same
/// messages) and generates constants that make using those messages in diagnostics more ergonomic.
///
/// For example, given the following invocation of the macro..
///
/// ```ignore (rust)
/// fluent_messages! {
///     typeck => "./typeck.ftl",
/// }
/// ```
/// ..where `typeck.ftl` has the following contents..
///
/// ```fluent
/// typeck_field_multiply_specified_in_initializer =
///     field `{$ident}` specified more than once
///     .label = used more than once
///     .label_previous_use = first use of `{$ident}`
/// ```
/// ...then the macro parse the Fluent resource, emitting a diagnostic if it fails to do so, and
/// will generate the following code:
///
/// ```ignore (rust)
/// pub static DEFAULT_LOCALE_RESOURCES: &'static [&'static str] = &[
///     include_str!("./typeck.ftl"),
/// ];
///
/// mod fluent_generated {
///     mod typeck {
///         pub const field_multiply_specified_in_initializer: DiagnosticMessage =
///             DiagnosticMessage::fluent("typeck_field_multiply_specified_in_initializer");
///         pub const field_multiply_specified_in_initializer_label_previous_use: DiagnosticMessage =
///             DiagnosticMessage::fluent_attr(
///                 "typeck_field_multiply_specified_in_initializer",
///                 "previous_use_label"
///             );
///     }
/// }
/// ```
/// When emitting a diagnostic, the generated constants can be used as follows:
///
/// ```ignore (rust)
/// let mut err = sess.struct_span_err(
///     span,
///     fluent::typeck::field_multiply_specified_in_initializer
/// );
/// err.span_default_label(span);
/// err.span_label(
///     previous_use_span,
///     fluent::typeck::field_multiply_specified_in_initializer_label_previous_use
/// );
/// err.emit();
/// ```
#[proc_macro]
pub fn fluent_messages(input: TokenStream) -> TokenStream {
    diagnostics::fluent_messages(input)
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
decl_derive!([TypeVisitable, attributes(type_visitable)] => type_visitable::type_visitable_derive);
decl_derive!([Lift, attributes(lift)] => lift::lift_derive);
decl_derive!(
    [SessionDiagnostic, attributes(
        // struct attributes
        diag,
        help,
        note,
        warning,
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
    [LintDiagnostic, attributes(
        // struct attributes
        diag,
        help,
        note,
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
    [SessionSubdiagnostic, attributes(
        // struct/variant attributes
        label,
        help,
        note,
        warning,
        suggestion,
        suggestion_short,
        suggestion_hidden,
        suggestion_verbose,
        // field attributes
        skip_arg,
        primary_span,
        applicability)] => diagnostics::session_subdiagnostic_derive
);
