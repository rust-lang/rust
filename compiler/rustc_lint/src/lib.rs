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
//! rustc can also load external lint plugins, as is done for Clippy.
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

// tidy-alphabetical-start
#![allow(internal_features)]
#![doc(html_root_url = "https://doc.rust-lang.org/nightly/nightly-rustc/")]
#![doc(rust_logo)]
#![feature(array_windows)]
#![feature(box_patterns)]
#![feature(control_flow_enum)]
#![feature(extract_if)]
#![feature(if_let_guard)]
#![feature(iter_order_by)]
#![feature(let_chains)]
#![feature(rustc_attrs)]
#![feature(rustdoc_internals)]
#![feature(trait_upcasting)]
// tidy-alphabetical-end

mod async_closures;
mod async_fn_in_trait;
pub mod builtin;
mod context;
mod deref_into_dyn_supertrait;
mod drop_forget_useless;
mod early;
mod enum_intrinsics_non_enums;
mod errors;
mod expect;
mod for_loops_over_fallibles;
mod foreign_modules;
pub mod hidden_unicode_codepoints;
mod impl_trait_overcaptures;
mod internal;
mod invalid_from_utf8;
mod late;
mod let_underscore;
mod levels;
mod lints;
mod macro_expr_fragment_specifier_2024_migration;
mod map_unit_fn;
mod methods;
mod multiple_supertrait_upcastable;
mod non_ascii_idents;
mod non_fmt_panic;
mod non_local_def;
mod nonstandard_style;
mod noop_method_call;
mod opaque_hidden_inferred_bound;
mod pass_by_value;
mod passes;
mod precedence;
mod ptr_nulls;
mod redundant_semicolon;
mod reference_casting;
mod shadowed_into_iter;
mod traits;
mod types;
mod unit_bindings;
mod unused;

pub use shadowed_into_iter::{ARRAY_INTO_ITER, BOXED_SLICE_INTO_ITER};

use rustc_hir::def_id::LocalModDefId;
use rustc_middle::query::Providers;
use rustc_middle::ty::TyCtxt;

use async_closures::AsyncClosureUsage;
use async_fn_in_trait::AsyncFnInTrait;
use builtin::*;
use deref_into_dyn_supertrait::*;
use drop_forget_useless::*;
use enum_intrinsics_non_enums::EnumIntrinsicsNonEnums;
use for_loops_over_fallibles::*;
use hidden_unicode_codepoints::*;
use impl_trait_overcaptures::ImplTraitOvercaptures;
use internal::*;
use invalid_from_utf8::*;
use let_underscore::*;
use macro_expr_fragment_specifier_2024_migration::*;
use map_unit_fn::*;
use methods::*;
use multiple_supertrait_upcastable::*;
use non_ascii_idents::*;
use non_fmt_panic::NonPanicFmt;
use non_local_def::*;
use nonstandard_style::*;
use noop_method_call::*;
use opaque_hidden_inferred_bound::*;
use pass_by_value::*;
use precedence::*;
use ptr_nulls::*;
use redundant_semicolon::*;
use reference_casting::*;
use shadowed_into_iter::ShadowedIntoIter;
use traits::*;
use types::*;
use unit_bindings::*;
use unused::*;

#[rustfmt::skip]
pub use builtin::{MissingDoc, SoftLints};
pub use context::{CheckLintNameResult, FindLintError, LintStore};
pub use context::{EarlyContext, LateContext, LintContext};
pub use early::{check_ast_node, EarlyCheckNode};
pub use late::{check_crate, late_lint_mod, unerased_lint_store};
pub use passes::{EarlyLintPass, LateLintPass};
pub use rustc_session::lint::Level::{self, *};
pub use rustc_session::lint::{BufferedEarlyLint, FutureIncompatibleInfo, Lint, LintId};
pub use rustc_session::lint::{LintPass, LintVec};

rustc_fluent_macro::fluent_messages! { "../messages.ftl" }

pub fn provide(providers: &mut Providers) {
    levels::provide(providers);
    expect::provide(providers);
    foreign_modules::provide(providers);
    *providers = Providers { lint_mod, ..*providers };
}

fn lint_mod(tcx: TyCtxt<'_>, module_def_id: LocalModDefId) {
    late_lint_mod(tcx, module_def_id, BuiltinCombinedModuleLateLintPass::new());
}

early_lint_methods!(
    declare_combined_early_lint_pass,
    [
        pub BuiltinCombinedPreExpansionLintPass,
        [
            KeywordIdents: KeywordIdents,
        ]
    ]
);

early_lint_methods!(
    declare_combined_early_lint_pass,
    [
        pub BuiltinCombinedEarlyLintPass,
        [
            UnusedParens: UnusedParens::new(),
            UnusedBraces: UnusedBraces,
            UnusedImportBraces: UnusedImportBraces,
            UnsafeCode: UnsafeCode,
            SpecialModuleName: SpecialModuleName,
            AnonymousParameters: AnonymousParameters,
            EllipsisInclusiveRangePatterns: EllipsisInclusiveRangePatterns::default(),
            NonCamelCaseTypes: NonCamelCaseTypes,
            DeprecatedAttr: DeprecatedAttr::new(),
            WhileTrue: WhileTrue,
            NonAsciiIdents: NonAsciiIdents,
            HiddenUnicodeCodepoints: HiddenUnicodeCodepoints,
            IncompleteInternalFeatures: IncompleteInternalFeatures,
            RedundantSemicolons: RedundantSemicolons,
            UnusedDocComment: UnusedDocComment,
            Expr2024: Expr2024,
            Precedence: Precedence,
        ]
    ]
);

late_lint_methods!(
    declare_combined_late_lint_pass,
    [
        BuiltinCombinedModuleLateLintPass,
        [
            ForLoopsOverFallibles: ForLoopsOverFallibles,
            DerefIntoDynSupertrait: DerefIntoDynSupertrait,
            DropForgetUseless: DropForgetUseless,
            HardwiredLints: HardwiredLints,
            ImproperCTypesDeclarations: ImproperCTypesDeclarations,
            ImproperCTypesDefinitions: ImproperCTypesDefinitions,
            InvalidFromUtf8: InvalidFromUtf8,
            VariantSizeDifferences: VariantSizeDifferences,
            PathStatements: PathStatements,
            LetUnderscore: LetUnderscore,
            InvalidReferenceCasting: InvalidReferenceCasting,
            // Depends on referenced function signatures in expressions
            UnusedResults: UnusedResults,
            UnitBindings: UnitBindings,
            NonUpperCaseGlobals: NonUpperCaseGlobals,
            NonShorthandFieldPatterns: NonShorthandFieldPatterns,
            UnusedAllocation: UnusedAllocation,
            // Depends on types used in type definitions
            MissingCopyImplementations: MissingCopyImplementations,
            // Depends on referenced function signatures in expressions
            PtrNullChecks: PtrNullChecks,
            MutableTransmutes: MutableTransmutes,
            TypeAliasBounds: TypeAliasBounds,
            TrivialConstraints: TrivialConstraints,
            TypeLimits: TypeLimits::new(),
            NonSnakeCase: NonSnakeCase,
            InvalidNoMangleItems: InvalidNoMangleItems,
            // Depends on effective visibilities
            UnreachablePub: UnreachablePub,
            ExplicitOutlivesRequirements: ExplicitOutlivesRequirements,
            InvalidValue: InvalidValue,
            DerefNullPtr: DerefNullPtr,
            UnstableFeatures: UnstableFeatures,
            UngatedAsyncFnTrackCaller: UngatedAsyncFnTrackCaller,
            ShadowedIntoIter: ShadowedIntoIter,
            DropTraitConstraints: DropTraitConstraints,
            TemporaryCStringAsPtr: TemporaryCStringAsPtr,
            NonPanicFmt: NonPanicFmt,
            NoopMethodCall: NoopMethodCall,
            EnumIntrinsicsNonEnums: EnumIntrinsicsNonEnums,
            InvalidAtomicOrdering: InvalidAtomicOrdering,
            AsmLabels: AsmLabels,
            OpaqueHiddenInferredBound: OpaqueHiddenInferredBound,
            MultipleSupertraitUpcastable: MultipleSupertraitUpcastable,
            MapUnitFn: MapUnitFn,
            MissingDebugImplementations: MissingDebugImplementations,
            MissingDoc: MissingDoc,
            AsyncClosureUsage: AsyncClosureUsage,
            AsyncFnInTrait: AsyncFnInTrait,
            NonLocalDefinitions: NonLocalDefinitions::default(),
            ImplTraitOvercaptures: ImplTraitOvercaptures,
        ]
    ]
);

pub fn new_lint_store(internal_lints: bool) -> LintStore {
    let mut lint_store = LintStore::new();

    register_builtins(&mut lint_store);
    if internal_lints {
        register_internals(&mut lint_store);
    }

    lint_store
}

/// Tell the `LintStore` about all the built-in lints (the ones
/// defined in this crate and the ones defined in
/// `rustc_session::lint::builtin`).
fn register_builtins(store: &mut LintStore) {
    macro_rules! add_lint_group {
        ($name:expr, $($lint:ident),*) => (
            store.register_group(false, $name, None, vec![$(LintId::of($lint)),*]);
        )
    }

    store.register_lints(&BuiltinCombinedPreExpansionLintPass::get_lints());
    store.register_lints(&BuiltinCombinedEarlyLintPass::get_lints());
    store.register_lints(&BuiltinCombinedModuleLateLintPass::get_lints());
    store.register_lints(&foreign_modules::get_lints());

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
        UNUSED_MACRO_RULES,
        UNUSED_ALLOCATION,
        UNUSED_DOC_COMMENTS,
        UNUSED_EXTERN_CRATES,
        UNUSED_FEATURES,
        UNUSED_LABELS,
        UNUSED_PARENS,
        UNUSED_BRACES,
        REDUNDANT_SEMICOLONS,
        MAP_UNIT_FN
    );

    add_lint_group!("let_underscore", LET_UNDERSCORE_DROP, LET_UNDERSCORE_LOCK);

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

    add_lint_group!("keyword_idents", KEYWORD_IDENTS_2018, KEYWORD_IDENTS_2024);

    add_lint_group!(
        "refining_impl_trait",
        REFINING_IMPL_TRAIT_REACHABLE,
        REFINING_IMPL_TRAIT_INTERNAL
    );

    add_lint_group!("deprecated_safe", DEPRECATED_SAFE_2024);

    // Register renamed and removed lints.
    store.register_renamed("single_use_lifetime", "single_use_lifetimes");
    store.register_renamed("elided_lifetime_in_path", "elided_lifetimes_in_paths");
    store.register_renamed("bare_trait_object", "bare_trait_objects");
    store.register_renamed("unstable_name_collision", "unstable_name_collisions");
    store.register_renamed("unused_doc_comment", "unused_doc_comments");
    store.register_renamed("async_idents", "keyword_idents_2018");
    store.register_renamed("exceeding_bitshifts", "arithmetic_overflow");
    store.register_renamed("redundant_semicolon", "redundant_semicolons");
    store.register_renamed("overlapping_patterns", "overlapping_range_endpoints");
    store.register_renamed("disjoint_capture_migration", "rust_2021_incompatible_closure_captures");
    store.register_renamed("or_patterns_back_compat", "rust_2021_incompatible_or_patterns");
    store.register_renamed("non_fmt_panic", "non_fmt_panics");
    store.register_renamed("unused_tuple_struct_fields", "dead_code");
    store.register_renamed("static_mut_ref", "static_mut_refs");

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
    store.register_removed(
        "unsupported_naked_functions",
        "converted into hard error, see RFC 2972 \
         <https://github.com/rust-lang/rfcs/blob/master/text/2972-constrained-naked.md> for more information",
    );
    store.register_removed(
        "mutable_borrow_reservation_conflict",
        "now allowed, see issue #59159 \
         <https://github.com/rust-lang/rust/issues/59159> for more information",
    );
    store.register_removed(
        "const_err",
        "converted into hard error, see issue #71800 \
         <https://github.com/rust-lang/rust/issues/71800> for more information",
    );
    store.register_removed(
        "safe_packed_borrows",
        "converted into hard error, see issue #82523 \
         <https://github.com/rust-lang/rust/issues/82523> for more information",
    );
    store.register_removed(
        "unaligned_references",
        "converted into hard error, see issue #82523 \
         <https://github.com/rust-lang/rust/issues/82523> for more information",
    );
    store.register_removed(
        "private_in_public",
        "replaced with another group of lints, see RFC \
         <https://rust-lang.github.io/rfcs/2145-type-privacy.html> for more information",
    );
    store.register_removed(
        "invalid_alignment",
        "converted into hard error, see PR #104616 \
         <https://github.com/rust-lang/rust/pull/104616> for more information",
    );
    store.register_removed(
        "implied_bounds_entailment",
        "converted into hard error, see PR #117984 \
        <https://github.com/rust-lang/rust/pull/117984> for more information",
    );
    store.register_removed(
        "coinductive_overlap_in_coherence",
        "converted into hard error, see PR #118649 \
         <https://github.com/rust-lang/rust/pull/118649> for more information",
    );
    store.register_removed(
        "illegal_floating_point_literal_pattern",
        "no longer a warning, float patterns behave the same as `==`",
    );
    store.register_removed(
        "nontrivial_structural_match",
        "no longer needed, see RFC #3535 \
         <https://rust-lang.github.io/rfcs/3535-constants-in-patterns.html> for more information",
    );
    store.register_removed(
        "suspicious_auto_trait_impls",
        "no longer needed, see #93367 \
         <https://github.com/rust-lang/rust/issues/93367> for more information",
    );
    store.register_removed(
        "const_patterns_without_partial_eq",
        "converted into hard error, see RFC #3535 \
         <https://rust-lang.github.io/rfcs/3535-constants-in-patterns.html> for more information",
    );
    store.register_removed(
        "indirect_structural_match",
        "converted into hard error, see RFC #3535 \
         <https://rust-lang.github.io/rfcs/3535-constants-in-patterns.html> for more information",
    );
    store.register_removed(
        "pointer_structural_match",
        "converted into hard error, see RFC #3535 \
         <https://rust-lang.github.io/rfcs/3535-constants-in-patterns.html> for more information",
    );
    store.register_removed(
        "box_pointers",
        "it does not detect other kinds of allocations, and existed only for historical reasons",
    );
}

fn register_internals(store: &mut LintStore) {
    store.register_lints(&LintPassImpl::get_lints());
    store.register_early_pass(|| Box::new(LintPassImpl));
    store.register_lints(&DefaultHashTypes::get_lints());
    store.register_late_mod_pass(|_| Box::new(DefaultHashTypes));
    store.register_lints(&QueryStability::get_lints());
    store.register_late_mod_pass(|_| Box::new(QueryStability));
    store.register_lints(&ExistingDocKeyword::get_lints());
    store.register_late_mod_pass(|_| Box::new(ExistingDocKeyword));
    store.register_lints(&TyTyKind::get_lints());
    store.register_late_mod_pass(|_| Box::new(TyTyKind));
    store.register_lints(&TypeIr::get_lints());
    store.register_late_mod_pass(|_| Box::new(TypeIr));
    store.register_lints(&Diagnostics::get_lints());
    store.register_late_mod_pass(|_| Box::new(Diagnostics));
    store.register_lints(&BadOptAccess::get_lints());
    store.register_late_mod_pass(|_| Box::new(BadOptAccess));
    store.register_lints(&PassByValue::get_lints());
    store.register_late_mod_pass(|_| Box::new(PassByValue));
    store.register_lints(&SpanUseEqCtxt::get_lints());
    store.register_late_mod_pass(|_| Box::new(SpanUseEqCtxt));
    // FIXME(davidtwco): deliberately do not include `UNTRANSLATABLE_DIAGNOSTIC` and
    // `DIAGNOSTIC_OUTSIDE_OF_IMPL` here because `-Wrustc::internal` is provided to every crate and
    // these lints will trigger all of the time - change this once migration to diagnostic structs
    // and translation is completed
    store.register_group(
        false,
        "rustc::internal",
        None,
        vec![
            LintId::of(DEFAULT_HASH_TYPES),
            LintId::of(POTENTIAL_QUERY_INSTABILITY),
            LintId::of(USAGE_OF_TY_TYKIND),
            LintId::of(PASS_BY_VALUE),
            LintId::of(LINT_PASS_IMPL_WITHOUT_MACRO),
            LintId::of(USAGE_OF_QUALIFIED_TY),
            LintId::of(NON_GLOB_IMPORT_OF_TYPE_IR_INHERENT),
            LintId::of(EXISTING_DOC_KEYWORD),
            LintId::of(BAD_OPT_ACCESS),
            LintId::of(SPAN_USE_EQ_CTXT),
        ],
    );
}

#[cfg(test)]
mod tests;
