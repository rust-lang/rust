//! # Lints in the Rust compiler
//!
//! This currently only contains the definitions and implementations
//! of most of the lints that `rustc` supports directly, it does not
//! contain the infrastructure for defining/registering lints. That is
//! available in `rustc::lint` and `rustc_plugin` respectively.
//!
//! ## Note
//!
//! This API is completely unstable and subject to change.

#![doc(html_root_url = "https://doc.rust-lang.org/nightly/")]

#![cfg_attr(test, feature(test))]
#![feature(box_patterns)]
#![feature(box_syntax)]
#![feature(nll)]
#![feature(rustc_diagnostic_macros)]

#![recursion_limit="256"]

#![deny(rust_2018_idioms)]
#![deny(internal)]
#![deny(unused_lifetimes)]

#[macro_use]
extern crate rustc;

mod error_codes;
mod nonstandard_style;
pub mod builtin;
mod types;
mod unused;

use rustc::lint;
use rustc::lint::{EarlyContext, LateContext, LateLintPass, EarlyLintPass, LintPass, LintArray};
use rustc::lint::builtin::{
    BARE_TRAIT_OBJECTS,
    ABSOLUTE_PATHS_NOT_STARTING_WITH_CRATE,
    ELIDED_LIFETIMES_IN_PATHS,
    EXPLICIT_OUTLIVES_REQUIREMENTS,
    INTRA_DOC_LINK_RESOLUTION_FAILURE,
    MISSING_DOC_CODE_EXAMPLES,
    PRIVATE_DOC_TESTS,
    parser::ILL_FORMED_ATTRIBUTE_INPUT,
};
use rustc::session;
use rustc::hir;
use rustc::hir::def_id::DefId;
use rustc::ty::query::Providers;
use rustc::ty::TyCtxt;

use syntax::ast;
use syntax::edition::Edition;
use syntax_pos::Span;

use session::Session;
use lint::LintId;
use lint::FutureIncompatibleInfo;

use nonstandard_style::*;
use builtin::*;
use types::*;
use unused::*;
use rustc::lint::internal::*;

/// Useful for other parts of the compiler.
pub use builtin::SoftLints;

pub fn provide(providers: &mut Providers<'_>) {
    *providers = Providers {
        lint_mod,
        ..*providers
    };
}

fn lint_mod(tcx: TyCtxt<'_>, module_def_id: DefId) {
    lint::late_lint_mod(tcx, module_def_id, BuiltinCombinedModuleLateLintPass::new());
}

macro_rules! pre_expansion_lint_passes {
    ($macro:path, $args:tt) => (
        $macro!($args, [
            KeywordIdents: KeywordIdents,
            UnusedDocComment: UnusedDocComment,
        ]);
    )
}

macro_rules! early_lint_passes {
    ($macro:path, $args:tt) => (
        $macro!($args, [
            UnusedParens: UnusedParens,
            UnusedImportBraces: UnusedImportBraces,
            UnsafeCode: UnsafeCode,
            AnonymousParameters: AnonymousParameters,
            EllipsisInclusiveRangePatterns: EllipsisInclusiveRangePatterns::default(),
            NonCamelCaseTypes: NonCamelCaseTypes,
            DeprecatedAttr: DeprecatedAttr::new(),
        ]);
    )
}

macro_rules! declare_combined_early_pass {
    ([$name:ident], $passes:tt) => (
        early_lint_methods!(declare_combined_early_lint_pass, [pub $name, $passes]);
    )
}

pre_expansion_lint_passes!(declare_combined_early_pass, [BuiltinCombinedPreExpansionLintPass]);
early_lint_passes!(declare_combined_early_pass, [BuiltinCombinedEarlyLintPass]);

macro_rules! late_lint_passes {
    ($macro:path, $args:tt) => (
        $macro!($args, [
            // FIXME: Look into regression when this is used as a module lint
            // May Depend on constants elsewhere
            UnusedBrokenConst: UnusedBrokenConst,

            // Uses attr::is_used which is untracked, can't be an incremental module pass.
            UnusedAttributes: UnusedAttributes::new(),

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
        ]);
    )
}

macro_rules! late_lint_mod_passes {
    ($macro:path, $args:tt) => (
        $macro!($args, [
            HardwiredLints: HardwiredLints,
            WhileTrue: WhileTrue,
            ImproperCTypes: ImproperCTypes,
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

            PluginAsLibrary: PluginAsLibrary,

            // Depends on referenced function signatures in expressions
            MutableTransmutes: MutableTransmutes,

            // Depends on types of fields, checks if they implement Drop
            UnionsWithDropFields: UnionsWithDropFields,

            TypeAliasBounds: TypeAliasBounds,

            TrivialConstraints: TrivialConstraints,
            TypeLimits: TypeLimits::new(),

            NonSnakeCase: NonSnakeCase,
            InvalidNoMangleItems: InvalidNoMangleItems,

            // Depends on access levels
            UnreachablePub: UnreachablePub,

            ExplicitOutlivesRequirements: ExplicitOutlivesRequirements,
        ]);
    )
}

macro_rules! declare_combined_late_pass {
    ([$v:vis $name:ident], $passes:tt) => (
        late_lint_methods!(declare_combined_late_lint_pass, [$v $name, $passes], ['tcx]);
    )
}

// FIXME: Make a separate lint type which do not require typeck tables
late_lint_passes!(declare_combined_late_pass, [pub BuiltinCombinedLateLintPass]);

late_lint_mod_passes!(declare_combined_late_pass, [BuiltinCombinedModuleLateLintPass]);

/// Tell the `LintStore` about all the built-in lints (the ones
/// defined in this crate and the ones defined in
/// `rustc::lint::builtin`).
pub fn register_builtins(store: &mut lint::LintStore, sess: Option<&Session>) {
    macro_rules! add_lint_group {
        ($sess:ident, $name:expr, $($lint:ident),*) => (
            store.register_group($sess, false, $name, None, vec![$(LintId::of($lint)),*]);
        )
    }

    macro_rules! register_pass {
        ($method:ident, $constructor:expr, [$($args:expr),*]) => (
            store.$method(sess, false, false, $($args,)* box $constructor);
        )
    }

    macro_rules! register_passes {
        ([$method:ident, $args:tt], [$($passes:ident: $constructor:expr,)*]) => (
            $(
                register_pass!($method, $constructor, $args);
            )*
        )
    }

    if sess.map(|sess| sess.opts.debugging_opts.no_interleave_lints).unwrap_or(false) {
        pre_expansion_lint_passes!(register_passes, [register_pre_expansion_pass, []]);
        early_lint_passes!(register_passes, [register_early_pass, []]);
        late_lint_passes!(register_passes, [register_late_pass, [false]]);
        late_lint_mod_passes!(register_passes, [register_late_pass, [true]]);
    } else {
        store.register_pre_expansion_pass(
            sess,
            false,
            true,
            box BuiltinCombinedPreExpansionLintPass::new()
        );
        store.register_early_pass(sess, false, true, box BuiltinCombinedEarlyLintPass::new());
        store.register_late_pass(
            sess, false, true, true, box BuiltinCombinedModuleLateLintPass::new()
        );
        store.register_late_pass(
            sess, false, true, false, box BuiltinCombinedLateLintPass::new()
        );
    }

    add_lint_group!(sess,
                    "nonstandard_style",
                    NON_CAMEL_CASE_TYPES,
                    NON_SNAKE_CASE,
                    NON_UPPER_CASE_GLOBALS);

    add_lint_group!(sess,
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
                    UNUSED_PARENS);

    add_lint_group!(sess,
                    "rust_2018_idioms",
                    BARE_TRAIT_OBJECTS,
                    UNUSED_EXTERN_CRATES,
                    ELLIPSIS_INCLUSIVE_RANGE_PATTERNS,
                    ELIDED_LIFETIMES_IN_PATHS,
                    EXPLICIT_OUTLIVES_REQUIREMENTS

                    // FIXME(#52665, #47816) not always applicable and not all
                    // macros are ready for this yet.
                    // UNREACHABLE_PUB,

                    // FIXME macro crates are not up for this yet, too much
                    // breakage is seen if we try to encourage this lint.
                    // MACRO_USE_EXTERN_CRATE,
                    );

    add_lint_group!(sess,
                    "rustdoc",
                    INTRA_DOC_LINK_RESOLUTION_FAILURE,
                    MISSING_DOC_CODE_EXAMPLES,
                    PRIVATE_DOC_TESTS);

    // Guidelines for creating a future incompatibility lint:
    //
    // - Create a lint defaulting to warn as normal, with ideally the same error
    //   message you would normally give
    // - Add a suitable reference, typically an RFC or tracking issue. Go ahead
    //   and include the full URL, sort items in ascending order of issue numbers.
    // - Later, change lint to error
    // - Eventually, remove lint
    store.register_future_incompatible(sess, vec![
        FutureIncompatibleInfo {
            id: LintId::of(PRIVATE_IN_PUBLIC),
            reference: "issue #34537 <https://github.com/rust-lang/rust/issues/34537>",
            edition: None,
        },
        FutureIncompatibleInfo {
            id: LintId::of(PUB_USE_OF_PRIVATE_EXTERN_CRATE),
            reference: "issue #34537 <https://github.com/rust-lang/rust/issues/34537>",
            edition: None,
        },
        FutureIncompatibleInfo {
            id: LintId::of(PATTERNS_IN_FNS_WITHOUT_BODY),
            reference: "issue #35203 <https://github.com/rust-lang/rust/issues/35203>",
            edition: None,
        },
        FutureIncompatibleInfo {
            id: LintId::of(DUPLICATE_MACRO_EXPORTS),
            reference: "issue #35896 <https://github.com/rust-lang/rust/issues/35896>",
            edition: Some(Edition::Edition2018),
        },
        FutureIncompatibleInfo {
            id: LintId::of(KEYWORD_IDENTS),
            reference: "issue #49716 <https://github.com/rust-lang/rust/issues/49716>",
            edition: Some(Edition::Edition2018),
        },
        FutureIncompatibleInfo {
            id: LintId::of(SAFE_EXTERN_STATICS),
            reference: "issue #36247 <https://github.com/rust-lang/rust/issues/36247>",
            edition: None,
        },
        FutureIncompatibleInfo {
            id: LintId::of(INVALID_TYPE_PARAM_DEFAULT),
            reference: "issue #36887 <https://github.com/rust-lang/rust/issues/36887>",
            edition: None,
        },
        FutureIncompatibleInfo {
            id: LintId::of(LEGACY_DIRECTORY_OWNERSHIP),
            reference: "issue #37872 <https://github.com/rust-lang/rust/issues/37872>",
            edition: None,
        },
        FutureIncompatibleInfo {
            id: LintId::of(LEGACY_CONSTRUCTOR_VISIBILITY),
            reference: "issue #39207 <https://github.com/rust-lang/rust/issues/39207>",
            edition: None,
        },
        FutureIncompatibleInfo {
            id: LintId::of(MISSING_FRAGMENT_SPECIFIER),
            reference: "issue #40107 <https://github.com/rust-lang/rust/issues/40107>",
            edition: None,
        },
        FutureIncompatibleInfo {
            id: LintId::of(ILLEGAL_FLOATING_POINT_LITERAL_PATTERN),
            reference: "issue #41620 <https://github.com/rust-lang/rust/issues/41620>",
            edition: None,
        },
        FutureIncompatibleInfo {
            id: LintId::of(ANONYMOUS_PARAMETERS),
            reference: "issue #41686 <https://github.com/rust-lang/rust/issues/41686>",
            edition: Some(Edition::Edition2018),
        },
        FutureIncompatibleInfo {
            id: LintId::of(PARENTHESIZED_PARAMS_IN_TYPES_AND_MODULES),
            reference: "issue #42238 <https://github.com/rust-lang/rust/issues/42238>",
            edition: None,
        },
        FutureIncompatibleInfo {
            id: LintId::of(LATE_BOUND_LIFETIME_ARGUMENTS),
            reference: "issue #42868 <https://github.com/rust-lang/rust/issues/42868>",
            edition: None,
        },
        FutureIncompatibleInfo {
            id: LintId::of(SAFE_PACKED_BORROWS),
            reference: "issue #46043 <https://github.com/rust-lang/rust/issues/46043>",
            edition: None,
        },
        FutureIncompatibleInfo {
            id: LintId::of(ORDER_DEPENDENT_TRAIT_OBJECTS),
            reference: "issue #56484 <https://github.com/rust-lang/rust/issues/56484>",
            edition: None,
        },
        FutureIncompatibleInfo {
            id: LintId::of(TYVAR_BEHIND_RAW_POINTER),
            reference: "issue #46906 <https://github.com/rust-lang/rust/issues/46906>",
            edition: Some(Edition::Edition2018),
        },
        FutureIncompatibleInfo {
            id: LintId::of(UNSTABLE_NAME_COLLISIONS),
            reference: "issue #48919 <https://github.com/rust-lang/rust/issues/48919>",
            edition: None,
            // Note: this item represents future incompatibility of all unstable functions in the
            //       standard library, and thus should never be removed or changed to an error.
        },
        FutureIncompatibleInfo {
            id: LintId::of(ABSOLUTE_PATHS_NOT_STARTING_WITH_CRATE),
            reference: "issue #53130 <https://github.com/rust-lang/rust/issues/53130>",
            edition: Some(Edition::Edition2018),
        },
        FutureIncompatibleInfo {
            id: LintId::of(WHERE_CLAUSES_OBJECT_SAFETY),
            reference: "issue #51443 <https://github.com/rust-lang/rust/issues/51443>",
            edition: None,
        },
        FutureIncompatibleInfo {
            id: LintId::of(PROC_MACRO_DERIVE_RESOLUTION_FALLBACK),
            reference: "issue #50504 <https://github.com/rust-lang/rust/issues/50504>",
            edition: None,
        },
        FutureIncompatibleInfo {
            id: LintId::of(MACRO_EXPANDED_MACRO_EXPORTS_ACCESSED_BY_ABSOLUTE_PATHS),
            reference: "issue #52234 <https://github.com/rust-lang/rust/issues/52234>",
            edition: None,
        },
        FutureIncompatibleInfo {
            id: LintId::of(ILL_FORMED_ATTRIBUTE_INPUT),
            reference: "issue #57571 <https://github.com/rust-lang/rust/issues/57571>",
            edition: None,
        },
        FutureIncompatibleInfo {
            id: LintId::of(AMBIGUOUS_ASSOCIATED_ITEMS),
            reference: "issue #57644 <https://github.com/rust-lang/rust/issues/57644>",
            edition: None,
        },
        FutureIncompatibleInfo {
            id: LintId::of(NESTED_IMPL_TRAIT),
            reference: "issue #59014 <https://github.com/rust-lang/rust/issues/59014>",
            edition: None,
        },
        FutureIncompatibleInfo {
            id: LintId::of(MUTABLE_BORROW_RESERVATION_CONFLICT),
            reference: "issue #59159 <https://github.com/rust-lang/rust/issues/59159>",
            edition: None,
        }
        ]);

    // Register renamed and removed lints.
    store.register_renamed("single_use_lifetime", "single_use_lifetimes");
    store.register_renamed("elided_lifetime_in_path", "elided_lifetimes_in_paths");
    store.register_renamed("bare_trait_object", "bare_trait_objects");
    store.register_renamed("unstable_name_collision", "unstable_name_collisions");
    store.register_renamed("unused_doc_comment", "unused_doc_comments");
    store.register_renamed("async_idents", "keyword_idents");
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
    store.register_removed("transmute_from_fn_item_types",
        "always cast functions before transmuting them");
    store.register_removed("hr_lifetime_in_assoc_type",
        "converted into hard error, see https://github.com/rust-lang/rust/issues/33685");
    store.register_removed("inaccessible_extern_crate",
        "converted into hard error, see https://github.com/rust-lang/rust/issues/36886");
    store.register_removed("super_or_self_in_global_path",
        "converted into hard error, see https://github.com/rust-lang/rust/issues/36888");
    store.register_removed("overlapping_inherent_impls",
        "converted into hard error, see https://github.com/rust-lang/rust/issues/36889");
    store.register_removed("illegal_floating_point_constant_pattern",
        "converted into hard error, see https://github.com/rust-lang/rust/issues/36890");
    store.register_removed("illegal_struct_or_enum_constant_pattern",
        "converted into hard error, see https://github.com/rust-lang/rust/issues/36891");
    store.register_removed("lifetime_underscore",
        "converted into hard error, see https://github.com/rust-lang/rust/issues/36892");
    store.register_removed("extra_requirement_in_impl",
        "converted into hard error, see https://github.com/rust-lang/rust/issues/37166");
    store.register_removed("legacy_imports",
        "converted into hard error, see https://github.com/rust-lang/rust/issues/38260");
    store.register_removed("coerce_never",
        "converted into hard error, see https://github.com/rust-lang/rust/issues/48950");
    store.register_removed("resolve_trait_on_defaulted_unit",
        "converted into hard error, see https://github.com/rust-lang/rust/issues/48950");
    store.register_removed("private_no_mangle_fns",
        "no longer a warning, #[no_mangle] functions always exported");
    store.register_removed("private_no_mangle_statics",
        "no longer a warning, #[no_mangle] statics always exported");
    store.register_removed("bad_repr",
        "replaced with a generic attribute input check");
    store.register_removed("duplicate_matcher_binding_name",
        "converted into hard error, see https://github.com/rust-lang/rust/issues/57742");
    store.register_removed("incoherent_fundamental_impls",
        "converted into hard error, see https://github.com/rust-lang/rust/issues/46205");
}

pub fn register_internals(store: &mut lint::LintStore, sess: Option<&Session>) {
    store.register_early_pass(sess, false, false, box DefaultHashTypes::new());
    store.register_late_pass(sess, false, false, false, box TyTyKind);
    store.register_group(
        sess,
        false,
        "internal",
        None,
        vec![
            LintId::of(DEFAULT_HASH_TYPES),
            LintId::of(USAGE_OF_TY_TYKIND),
            LintId::of(TY_PASS_BY_REFERENCE),
            LintId::of(USAGE_OF_QUALIFIED_TY),
        ],
    );
}
