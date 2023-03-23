#![feature(allow_internal_unstable)]
#![feature(if_let_guard)]
#![feature(let_chains)]
#![feature(never_type)]
#![feature(proc_macro_diagnostic)]
#![feature(proc_macro_span)]
#![feature(proc_macro_tracked_env)]
#![allow(rustc::default_hash_types)]
#![deny(rustc::untranslatable_diagnostic)]
#![deny(rustc::diagnostic_outside_of_impl)]
#![allow(internal_features)]
#![recursion_limit = "128"]

use synstructure::decl_derive;

use proc_macro::TokenStream;

mod current_version;
mod diagnostics;
mod hash_stable;
mod lift;
mod newtype;
mod query;
mod serialize;
mod symbols;
mod traversable;

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
#[allow_internal_unstable(step_trait, rustc_attrs, trusted_step, spec_option_partial_eq)]
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
decl_derive!(
    [TypeFoldable, attributes(skip_traversal)] =>
    /// Derives `TypeFoldable` for the annotated `struct` or `enum` (`union` is not supported).
    ///
    /// Folds will produce a value of the same struct or enum variant as the input, with trivial
    /// fields unchanged and all non-trivial fields respectively folded (in definition order) using
    /// the `TypeFoldable` implementation for its type. A field of type `T` is "trivial" if `T`
    /// both does not reference any generic type parameters and either
    /// - does not reference any `'tcx` lifetime parameter; or
    /// - does not contain anything that could be of interest to folders.
    ///
    /// Non-trivial fields (e.g. of type `T`) can instead be left unchanged by applying
    /// `#[skip_traversal(because_trivial)]` to the field definition (or even to a variant
    /// definition if it should apply to all fields therein), but the derived implementation will
    /// only be applicable if `T` does not contain anything that may be of interest to folders
    /// (thus preventing fields from being so-skipped erroneously).
    ///
    /// In some rare situations, it may be desirable to skip folding of an item or field (or
    /// variant) that might otherwise be of interest to folders: **this is dangerous and could lead
    /// to miscompilation if user expectations are not met!**  Nevertheless, such can be achieved
    /// via a `#[skip_traversal(despite_potential_miscompilation_because = "<reason>"]` attribute.
    ///
    /// If the annotated type has a `'tcx` lifetime parameter, then that will be used as the
    /// lifetime for the type context/interner; otherwise the lifetime of the type context/interner
    /// will be unrelated to the annotated type. It therefore matters how any lifetime parameters of
    /// the annotated type are named. For example, deriving `TypeFoldable` for both `Foo<'a>` and
    /// `Bar<'tcx>` will respectively produce:
    ///
    /// `impl<'a, 'tcx> TypeFoldable<TyCtxt<'tcx>> for Foo<'a>`
    ///
    /// and
    ///
    /// `impl<'tcx> TypeFoldable<TyCtxt<'tcx>> for Bar<'tcx>`
    traversable::traversable_derive::<traversable::Foldable>
);
decl_derive!(
    [TypeVisitable, attributes(skip_traversal)] =>
    /// Derives `TypeVisitable` for the annotated `struct` or `enum` (`union` is not supported).
    ///
    /// Each non-trivial field of the struct or enum variant will be visited (in definition order)
    /// using the `TypeVisitable` implementation for its type; trivial fields will be ignored. A
    /// field of type `T` is "trivial" if `T` both does not reference any generic type parameters
    /// and either
    /// - does not reference any `'tcx` lifetime parameter; or
    /// - does not contain anything that could be of interest to visitors.
    ///
    /// Non-trivial fields (e.g. of type `T`) can instead be ignored by applying
    /// `#[skip_traversal(because_trivial)]` to the field definition (or even to a variant
    /// definition if it should apply to all fields therein), but the derived implementation will
    /// only be applicable if `T` does not contain anything that may be of interest to visitors
    /// (thus preventing fields from being so-skipped erroneously).
    ///
    /// In some rare situations, it may be desirable to skip visiting of an item or field (or
    /// variant) that might otherwise be of interest to visitors: **this is dangerous and could lead
    /// to miscompilation if user expectations are not met!**  Nevertheless, such can be achieved
    /// via a `#[skip_traversal(despite_potential_miscompilation_because = "<reason>"]` attribute.
    ///
    /// If the annotated type has a `'tcx` lifetime parameter, then that will be used as the
    /// lifetime for the type context/interner; otherwise the lifetime of the type context/interner
    /// will be unrelated to the annotated type. It therefore matters how any lifetime parameters of
    /// the annotated type are named. For example, deriving `TypeVisitable` for both `Foo<'a>` and
    /// `Bar<'tcx>` will respectively produce:
    ///
    /// `impl<'a, 'tcx> TypeVisitable<TyCtxt<'tcx>> for Foo<'a>`
    ///
    /// and
    ///
    /// `impl<'tcx> TypeVisitable<TyCtxt<'tcx>> for Bar<'tcx>`
    traversable::traversable_derive::<traversable::Visitable>
);
decl_derive!([Lift, attributes(lift)] => lift::lift_derive);
decl_derive!(
    [Diagnostic, attributes(
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
    [Subdiagnostic, attributes(
        // struct/variant attributes
        label,
        help,
        note,
        warning,
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
        applicability)] => diagnostics::session_subdiagnostic_derive
);
