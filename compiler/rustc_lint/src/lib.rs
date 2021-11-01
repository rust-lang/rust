//! Lints, aka compiler warnings.
//!
//! A 'lint' check is a kind of miscellaneous constraint that a user _might_
//! want to enforce, but might reasonably want to permit as well, on a
//! module-by-module basis. They contrast with static constraints enforced by
//! other phases of the compiler, which are generally required to hold in order
//! to compile the program at all.
//!
//! Most lints can be written as [LintPass] instances. These run after
//! all other analyses. The `LintPass`es built into rustc are defined
//! within [rustc_session::lint::builtin],
//! which has further comments on how to add such a lint.
//! rustc can also load user-defined lint plugins via the plugin mechanism.
//!
//! Some of rustc's lints are defined elsewhere in the compiler and work by
//! calling `add_lint()` on the overall `Session` object. This works when
//! it happens before the main lint pass, which emits the lints stored by
//! `add_lint()`. To emit lints after the main lint pass (from codegen, for
//! example) requires more effort. See `emit_lint` and `GatherNodeLevels`
//! in `context.rs`.
//!
//! Some code also exists in [rustc_session::lint], [rustc_middle::lint].
//!
//! ## Note
//!
//! This API is completely unstable and subject to change.

#![doc(html_root_url = "https://doc.rust-lang.org/nightly/nightly-rustc/")]
#![feature(array_windows)]
#![feature(bool_to_option)]
#![feature(box_patterns)]
#![feature(crate_visibility_modifier)]
#![feature(format_args_capture)]
#![feature(iter_order_by)]
#![feature(iter_zip)]
#![feature(never_type)]
#![feature(nll)]
#![feature(control_flow_enum)]
#![recursion_limit = "256"]

#[macro_use]
extern crate rustc_middle;
#[macro_use]
extern crate rustc_session;

mod array_into_iter;
pub mod builtin;
mod context;
mod early;
mod enum_intrinsics_non_enums;
pub mod hidden_unicode_codepoints;
mod internal;
mod late;
mod levels;
mod methods;
mod non_ascii_idents;
mod non_fmt_panic;
mod nonstandard_style;
mod noop_method_call;
mod passes;
mod redundant_semicolon;
mod traits;
mod types;
mod unused;

pub use array_into_iter::ARRAY_INTO_ITER;

use rustc_ast as ast;
use rustc_hir as hir;
use rustc_hir::def_id::LocalDefId;
use rustc_middle::ty::query::Providers;
use rustc_middle::ty::TyCtxt;
use rustc_session::lint::builtin::{
    BARE_TRAIT_OBJECTS, ELIDED_LIFETIMES_IN_PATHS, EXPLICIT_OUTLIVES_REQUIREMENTS,
};
use rustc_span::symbol::{Ident, Symbol};
use rustc_span::Span;

use array_into_iter::ArrayIntoIter;
use builtin::*;
use enum_intrinsics_non_enums::EnumIntrinsicsNonEnums;
use hidden_unicode_codepoints::*;
use internal::*;
use methods::*;
use non_ascii_idents::*;
use non_fmt_panic::NonPanicFmt;
use nonstandard_style::*;
use noop_method_call::*;
use redundant_semicolon::*;
use traits::*;
use types::*;
use unused::*;

/// Useful for other parts of the compiler / Clippy.
pub use builtin::SoftLints;
pub use context::{CheckLintNameResult, EarlyContext, LateContext, LintContext, LintStore};
pub use early::check_ast_crate;
pub use late::check_crate;
pub use passes::{EarlyLintPass, LateLintPass};
pub use rustc_session::lint::Level::{self, *};
pub use rustc_session::lint::{BufferedEarlyLint, FutureIncompatibleInfo, Lint, LintId};
pub use rustc_session::lint::{LintArray, LintPass};

pub fn provide(providers: &mut Providers) {
    levels::provide(providers);
    *providers = Providers { lint_mod, ..*providers };
}

fn lint_mod(tcx: TyCtxt<'_>, module_def_id: LocalDefId) {
    late::late_lint_mod(tcx, module_def_id, BuiltinCombinedModuleLateLintPass::new());
}

macro_rules! pre_expansion_lint_passes {
    ($macro:path, $args:tt) => {
        $macro!($args, [KeywordIdents: KeywordIdents,]);
    };
}

macro_rules! early_lint_passes {
    ($macro:path, $args:tt) => {
        $macro!(
            $args,
            [
                UnusedParens: UnusedParens,
                UnusedBraces: UnusedBraces,
                UnusedImportBraces: UnusedImportBraces,
                UnsafeCode: UnsafeCode,
                AnonymousParameters: AnonymousParameters,
                EllipsisInclusiveRangePatterns: EllipsisInclusiveRangePatterns::default(),
                NonCamelCaseTypes: NonCamelCaseTypes,
                DeprecatedAttr: DeprecatedAttr::new(),
                WhileTrue: WhileTrue,
                NonAsciiIdents: NonAsciiIdents,
                HiddenUnicodeCodepoints: HiddenUnicodeCodepoints,
                IncompleteFeatures: IncompleteFeatures,
                RedundantSemicolons: RedundantSemicolons,
                UnusedDocComment: UnusedDocComment,
            ]
        );
    };
}

macro_rules! declare_combined_early_pass {
    ([$name:ident], $passes:tt) => (
        early_lint_methods!(declare_combined_early_lint_pass, [pub $name, $passes]);
    )
}

pre_expansion_lint_passes!(declare_combined_early_pass, [BuiltinCombinedPreExpansionLintPass]);
early_lint_passes!(declare_combined_early_pass, [BuiltinCombinedEarlyLintPass]);

macro_rules! late_lint_passes {
    ($macro:path, $args:tt) => {
        $macro!(
            $args,
            [
                // FIXME: Look into regression when this is used as a module lint
                // May Depend on constants elsewhere
                UnusedBrokenConst: UnusedBrokenConst,
                // Needs to run after UnusedAttributes as it marks all `feature` attributes as used.
                UnstableFeatures: UnstableFeatures,
                // Tracks state across modules
                UnnameableTestItems: UnnameableTestItems::new(),
                // Tracks attributes of parents
                MissingDoc: MissingDoc::new(),
                // Depends on access levels
                // FIXME: Turn the computation of types which implement Debug into a query
                // and change this to a module lint pass
                MissingDebugImplementations: MissingDebugImplementations::default(),
                ArrayIntoIter: ArrayIntoIter::default(),
                ClashingExternDeclarations: ClashingExternDeclarations::new(),
                DropTraitConstraints: DropTraitConstraints,
                TemporaryCStringAsPtr: TemporaryCStringAsPtr,
                NonPanicFmt: NonPanicFmt,
                NoopMethodCall: NoopMethodCall,
                EnumIntrinsicsNonEnums: EnumIntrinsicsNonEnums,
                InvalidAtomicOrdering: InvalidAtomicOrdering,
                NamedAsmLabels: NamedAsmLabels,
            ]
        );
    };
}

macro_rules! late_lint_mod_passes {
    ($macro:path, $args:tt) => {
        $macro!(
            $args,
            [
                HardwiredLints: HardwiredLints,
                ImproperCTypesDeclarations: ImproperCTypesDeclarations,
                ImproperCTypesDefinitions: ImproperCTypesDefinitions,
                VariantSizeDifferences: VariantSizeDifferences,
                BoxPointers: BoxPointers,
                PathStatements: PathStatements,
                // Depends on referenced function signatures in expressions
                UnusedResults: UnusedResults,
                NonUpperCaseGlobals: NonUpperCaseGlobals,
                NonShorthandFieldPatterns: NonShorthandFieldPatterns,
                UnusedAllocation: UnusedAllocation,
                // Depends on types used in type definitions
                MissingCopyImplementations: MissingCopyImplementations,
                // Depends on referenced function signatures in expressions
                MutableTransmutes: MutableTransmutes,
                TypeAliasBounds: TypeAliasBounds,
                TrivialConstraints: TrivialConstraints,
                TypeLimits: TypeLimits::new(),
                NonSnakeCase: NonSnakeCase,
                InvalidNoMangleItems: InvalidNoMangleItems,
                // Depends on access levels
                UnreachablePub: UnreachablePub,
                ExplicitOutlivesRequirements: ExplicitOutlivesRequirements,
                InvalidValue: InvalidValue,
                DerefNullPtr: DerefNullPtr,
            ]
        );
    };
}

macro_rules! declare_combined_late_pass {
    ([$v:vis $name:ident], $passes:tt) => (
        late_lint_methods!(declare_combined_late_lint_pass, [$v $name, $passes], ['tcx]);
    )
}

// FIXME: Make a separate lint type which do not require typeck tables
late_lint_passes!(declare_combined_late_pass, [pub BuiltinCombinedLateLintPass]);

late_lint_mod_passes!(declare_combined_late_pass, [BuiltinCombinedModuleLateLintPass]);

pub fn new_lint_store(no_interleave_lints: bool, internal_lints: bool) -> LintStore {
    let mut lint_store = LintStore::new();

    register_builtins(&mut lint_store, no_interleave_lints);
    if internal_lints {
        register_internals(&mut lint_store);
    }

    lint_store
}

/// Tell the `LintStore` about all the built-in lints (the ones
/// defined in this crate and the ones defined in
/// `rustc_session::lint::builtin`).
fn register_builtins(store: &mut LintStore, no_interleave_lints: bool) {
    macro_rules! add_lint_group {
        ($name:expr, $($lint:ident),*) => (
            store.register_group(false, $name, None, vec![$(LintId::of($lint)),*]);
        )
    }

    macro_rules! register_pass {
        ($method:ident, $ty:ident, $constructor:expr) => {
            store.register_lints(&$ty::get_lints());
            store.$method(|| Box::new($constructor));
        };
    }

    macro_rules! register_passes {
        ($method:ident, [$($passes:ident: $constructor:expr,)*]) => (
            $(
                register_pass!($method, $passes, $constructor);
            )*
        )
    }

    if no_interleave_lints {
        pre_expansion_lint_passes!(register_passes, register_pre_expansion_pass);
        early_lint_passes!(register_passes, register_early_pass);
        late_lint_passes!(register_passes, register_late_pass);
        late_lint_mod_passes!(register_passes, register_late_mod_pass);
    } else {
        store.register_lints(&BuiltinCombinedPreExpansionLintPass::get_lints());
        store.register_lints(&BuiltinCombinedEarlyLintPass::get_lints());
        store.register_lints(&BuiltinCombinedModuleLateLintPass::get_lints());
        store.register_lints(&BuiltinCombinedLateLintPass::get_lints());
    }

    add_lint_group!(
        "nonstandard_style",
        NON_CAMEL_CASE_TYPES,
        NON_SNAKE_CASE,
        NON_UPPER_CASE_GLOBALS
    );

    add_lint_group!(
        "unused",
        UNUSED_IMPORTS,
        UNUSED_VARIABLES,
        UNUSED_ASSIGNMENTS,
        DEAD_CODE,
        UNUSED_MUT,
        UNREACHABLE_CODE,
        UNREACHABLE_PATTERNS,
        UNUSED_MUST_USE,
        UNUSED_UNSAFE,
        PATH_STATEMENTS,
        UNUSED_ATTRIBUTES,
        UNUSED_MACROS,
        UNUSED_ALLOCATION,
        UNUSED_DOC_COMMENTS,
        UNUSED_EXTERN_CRATES,
        UNUSED_FEATURES,
        UNUSED_LABELS,
        UNUSED_PARENS,
        UNUSED_BRACES,
        REDUNDANT_SEMICOLONS
    );

    add_lint_group!(
        "rust_2018_idioms",
        BARE_TRAIT_OBJECTS,
        UNUSED_EXTERN_CRATES,
        ELLIPSIS_INCLUSIVE_RANGE_PATTERNS,
        ELIDED_LIFETIMES_IN_PATHS,
        EXPLICIT_OUTLIVES_REQUIREMENTS // FIXME(#52665, #47816) not always applicable and not all
                                       // macros are ready for this yet.
                                       // UNREACHABLE_PUB,

                                       // FIXME macro crates are not up for this yet, too much
                                       // breakage is seen if we try to encourage this lint.
                                       // MACRO_USE_EXTERN_CRATE
    );

    // Register renamed and removed lints.
    store.register_renamed("single_use_lifetime", "single_use_lifetimes");
    store.register_renamed("elided_lifetime_in_path", "elided_lifetimes_in_paths");
    store.register_renamed("bare_trait_object", "bare_trait_objects");
    store.register_renamed("unstable_name_collision", "unstable_name_collisions");
    store.register_renamed("unused_doc_comment", "unused_doc_comments");
    store.register_renamed("async_idents", "keyword_idents");
    store.register_renamed("exceeding_bitshifts", "arithmetic_overflow");
    store.register_renamed("redundant_semicolon", "redundant_semicolons");
    store.register_renamed("overlapping_patterns", "overlapping_range_endpoints");
    store.register_renamed("safe_packed_borrows", "unaligned_references");
    store.register_renamed("disjoint_capture_migration", "rust_2021_incompatible_closure_captures");
    store.register_renamed("or_patterns_back_compat", "rust_2021_incompatible_or_patterns");
    store.register_renamed("non_fmt_panic", "non_fmt_panics");

    // These were moved to tool lints, but rustc still sees them when compiling normally, before
    // tool lints are registered, so `check_tool_name_for_backwards_compat` doesn't work. Use
    // `register_removed` explicitly.
    const RUSTDOC_LINTS: &[&str] = &[
        "broken_intra_doc_links",
        "private_intra_doc_links",
        "missing_crate_level_docs",
        "missing_doc_code_examples",
        "private_doc_tests",
        "invalid_codeblock_attributes",
        "invalid_html_tags",
        "non_autolinks",
    ];
    for rustdoc_lint in RUSTDOC_LINTS {
        store.register_ignored(rustdoc_lint);
    }
    store.register_removed(
        "intra_doc_link_resolution_failure",
        "use `rustdoc::broken_intra_doc_links` instead",
    );
    store.register_removed("rustdoc", "use `rustdoc::all` instead");

    store.register_removed("unknown_features", "replaced by an error");
    store.register_removed("unsigned_negation", "replaced by negate_unsigned feature gate");
    store.register_removed("negate_unsigned", "cast a signed value instead");
    store.register_removed("raw_pointer_derive", "using derive with raw pointers is ok");
    // Register lint group aliases.
    store.register_group_alias("nonstandard_style", "bad_style");
    // This was renamed to `raw_pointer_derive`, which was then removed,
    // so it is also considered removed.
    store.register_removed("raw_pointer_deriving", "using derive with raw pointers is ok");
    store.register_removed("drop_with_repr_extern", "drop flags have been removed");
    store.register_removed("fat_ptr_transmutes", "was accidentally removed back in 2014");
    store.register_removed("deprecated_attr", "use `deprecated` instead");
    store.register_removed(
        "transmute_from_fn_item_types",
        "always cast functions before transmuting them",
    );
    store.register_removed(
        "hr_lifetime_in_assoc_type",
        "converted into hard error, see issue #33685 \
         <https://github.com/rust-lang/rust/issues/33685> for more information",
    );
    store.register_removed(
        "inaccessible_extern_crate",
        "converted into hard error, see issue #36886 \
         <https://github.com/rust-lang/rust/issues/36886> for more information",
    );
    store.register_removed(
        "super_or_self_in_global_path",
        "converted into hard error, see issue #36888 \
         <https://github.com/rust-lang/rust/issues/36888> for more information",
    );
    store.register_removed(
        "overlapping_inherent_impls",
        "converted into hard error, see issue #36889 \
         <https://github.com/rust-lang/rust/issues/36889> for more information",
    );
    store.register_removed(
        "illegal_floating_point_constant_pattern",
        "converted into hard error, see issue #36890 \
         <https://github.com/rust-lang/rust/issues/36890> for more information",
    );
    store.register_removed(
        "illegal_struct_or_enum_constant_pattern",
        "converted into hard error, see issue #36891 \
         <https://github.com/rust-lang/rust/issues/36891> for more information",
    );
    store.register_removed(
        "lifetime_underscore",
        "converted into hard error, see issue #36892 \
         <https://github.com/rust-lang/rust/issues/36892> for more information",
    );
    store.register_removed(
        "extra_requirement_in_impl",
        "converted into hard error, see issue #37166 \
         <https://github.com/rust-lang/rust/issues/37166> for more information",
    );
    store.register_removed(
        "legacy_imports",
        "converted into hard error, see issue #38260 \
         <https://github.com/rust-lang/rust/issues/38260> for more information",
    );
    store.register_removed(
        "coerce_never",
        "converted into hard error, see issue #48950 \
         <https://github.com/rust-lang/rust/issues/48950> for more information",
    );
    store.register_removed(
        "resolve_trait_on_defaulted_unit",
        "converted into hard error, see issue #48950 \
         <https://github.com/rust-lang/rust/issues/48950> for more information",
    );
    store.register_removed(
        "private_no_mangle_fns",
        "no longer a warning, `#[no_mangle]` functions always exported",
    );
    store.register_removed(
        "private_no_mangle_statics",
        "no longer a warning, `#[no_mangle]` statics always exported",
    );
    store.register_removed("bad_repr", "replaced with a generic attribute input check");
    store.register_removed(
        "duplicate_matcher_binding_name",
        "converted into hard error, see issue #57742 \
         <https://github.com/rust-lang/rust/issues/57742> for more information",
    );
    store.register_removed(
        "incoherent_fundamental_impls",
        "converted into hard error, see issue #46205 \
         <https://github.com/rust-lang/rust/issues/46205> for more information",
    );
    store.register_removed(
        "legacy_constructor_visibility",
        "converted into hard error, see issue #39207 \
         <https://github.com/rust-lang/rust/issues/39207> for more information",
    );
    store.register_removed(
        "legacy_directory_ownership",
        "converted into hard error, see issue #37872 \
         <https://github.com/rust-lang/rust/issues/37872> for more information",
    );
    store.register_removed(
        "safe_extern_statics",
        "converted into hard error, see issue #36247 \
         <https://github.com/rust-lang/rust/issues/36247> for more information",
    );
    store.register_removed(
        "parenthesized_params_in_types_and_modules",
        "converted into hard error, see issue #42238 \
         <https://github.com/rust-lang/rust/issues/42238> for more information",
    );
    store.register_removed(
        "duplicate_macro_exports",
        "converted into hard error, see issue #35896 \
         <https://github.com/rust-lang/rust/issues/35896> for more information",
    );
    store.register_removed(
        "nested_impl_trait",
        "converted into hard error, see issue #59014 \
         <https://github.com/rust-lang/rust/issues/59014> for more information",
    );
    store.register_removed("plugin_as_library", "plugins have been deprecated and retired");
}

fn register_internals(store: &mut LintStore) {
    store.register_lints(&LintPassImpl::get_lints());
    store.register_early_pass(|| Box::new(LintPassImpl));
    store.register_lints(&DefaultHashTypes::get_lints());
    store.register_late_pass(|| Box::new(DefaultHashTypes));
    store.register_lints(&ExistingDocKeyword::get_lints());
    store.register_late_pass(|| Box::new(ExistingDocKeyword));
    store.register_lints(&TyTyKind::get_lints());
    store.register_late_pass(|| Box::new(TyTyKind));
    store.register_group(
        false,
        "rustc::internal",
        None,
        vec![
            LintId::of(DEFAULT_HASH_TYPES),
            LintId::of(USAGE_OF_TY_TYKIND),
            LintId::of(LINT_PASS_IMPL_WITHOUT_MACRO),
            LintId::of(TY_PASS_BY_REFERENCE),
            LintId::of(USAGE_OF_QUALIFIED_TY),
            LintId::of(EXISTING_DOC_KEYWORD),
        ],
    );
}

#[cfg(test)]
mod tests;
