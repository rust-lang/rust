#![feature(allow_internal_unstable)]
#![feature(if_let_guard)]
#![feature(let_chains)]
#![feature(never_type)]
#![feature(proc_macro_diagnostic)]
#![feature(proc_macro_span)]
#![feature(proc_macro_tracked_env)]
#![allow(rustc::default_hash_types)]
#![allow(internal_features)]

use synstructure::decl_derive;

use proc_macro::TokenStream;

mod current_version;
mod diagnostics;
mod hash_stable;
mod lift;
mod query;
mod serialize;
mod symbols;
mod traversable;

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

decl_derive!([Decodable_Generic] => serialize::decodable_generic_derive);
decl_derive!([Encodable_Generic] => serialize::encodable_generic_derive);
decl_derive!([Decodable] => serialize::decodable_derive);
decl_derive!([Encodable] => serialize::encodable_derive);
decl_derive!([TyDecodable] => serialize::type_decodable_derive);
decl_derive!([TyEncodable] => serialize::type_encodable_derive);
decl_derive!([MetadataDecodable] => serialize::meta_decodable_derive);
decl_derive!([MetadataEncodable] => serialize::meta_encodable_derive);
decl_derive!(
    [TypeFoldable, attributes(type_foldable, inline_traversals)] =>
    /// Derives `TypeFoldable` for the annotated `struct` or `enum` (`union` is not supported).
    ///
    /// Folds will produce a value of the same struct or enum variant as the input, with each field
    /// respectively folded (in definition order) using the `TypeFoldable` implementation for its
    /// type. However, if a field of a struct or of an enum variant is annotated with
    /// `#[type_foldable(identity)]` then that field will retain its incumbent value (and its type
    /// is not required to implement `TypeFoldable`). However use of this attribute is dangerous
    /// and should be used with extreme caution: should the type of the annotated field contain
    /// (now or in the future) a type that is of interest to a folder, it will not get folded (which
    /// may result in unexpected, hard-to-track bugs that could result in unsoundness).
    ///
    /// If the annotated item has a `'tcx` lifetime parameter, then that will be used as the
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
    ///
    /// The annotated item may be decorated with an `#[inline_traversals]` attribute to cause the
    /// generated folding method to be marked `#[inline]`.
    traversable::traversable_derive::<traversable::Foldable>
);
decl_derive!(
    [TypeVisitable, attributes(type_visitable, inline_traversals)] =>
    /// Derives `TypeVisitable` for the annotated `struct` or `enum` (`union` is not supported).
    ///
    /// Each field of the struct or enum variant will be visited (in definition order) using the
    /// `TypeVisitable` implementation for its type. However, if a field of a struct or of an enum
    /// variant is annotated with `#[type_visitable(ignore)]` then that field will not be visited
    /// (and its type is not required to implement `TypeVisitable`). However use of this attribute
    /// is dangerous and should be used with extreme caution: should the type of the annotated
    /// field (now or in the future) a type that is of interest to a visitor, it will not get
    /// visited (which may result in unexpected, hard-to-track bugs that could result in
    /// unsoundness).
    ///
    /// If the annotated item has a `'tcx` lifetime parameter, then that will be used as the
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
    ///
    /// The annotated item may be decorated with an `#[inline_traversals]` attribute to cause the
    /// generated folding method to be marked `#[inline]`.
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
