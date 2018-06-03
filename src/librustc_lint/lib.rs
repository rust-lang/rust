// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Lints in the Rust compiler.
//!
//! This currently only contains the definitions and implementations
//! of most of the lints that `rustc` supports directly, it does not
//! contain the infrastructure for defining/registering lints. That is
//! available in `rustc::lint` and `rustc_plugin` respectively.
//!
//! # Note
//!
//! This API is completely unstable and subject to change.

#![doc(html_logo_url = "https://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
      html_favicon_url = "https://doc.rust-lang.org/favicon.ico",
      html_root_url = "https://doc.rust-lang.org/nightly/")]

#![cfg_attr(test, feature(test))]
#![feature(box_patterns)]
#![feature(box_syntax)]
#![feature(macro_vis_matcher)]
#![feature(quote)]
#![feature(rustc_diagnostic_macros)]

extern crate syntax;
#[macro_use]
extern crate rustc;
#[macro_use]
extern crate log;
extern crate rustc_mir;
extern crate rustc_target;
extern crate syntax_pos;

use rustc::lint;
use rustc::lint::builtin::{BARE_TRAIT_OBJECTS, ABSOLUTE_PATHS_NOT_STARTING_WITH_CRATE};
use rustc::session;
use rustc::util;

use session::Session;
use syntax::edition::Edition;
use lint::LintId;
use lint::FutureIncompatibleInfo;

mod bad_style;
mod builtin;
mod types;
mod unused;

use bad_style::*;
use builtin::*;
use types::*;
use unused::*;

/// Tell the `LintStore` about all the built-in lints (the ones
/// defined in this crate and the ones defined in
/// `rustc::lint::builtin`).
pub fn register_builtins(store: &mut lint::LintStore, sess: Option<&Session>) {
    macro_rules! add_builtin {
        ($sess:ident, $($name:ident),*,) => (
            {$(
                store.register_late_pass($sess, false, box $name);
                )*}
            )
    }

    macro_rules! add_early_builtin {
        ($sess:ident, $($name:ident),*,) => (
            {$(
                store.register_early_pass($sess, false, box $name);
                )*}
            )
    }

    macro_rules! add_builtin_with_new {
        ($sess:ident, $($name:ident),*,) => (
            {$(
                store.register_late_pass($sess, false, box $name::new());
                )*}
            )
    }

    macro_rules! add_early_builtin_with_new {
        ($sess:ident, $($name:ident),*,) => (
            {$(
                store.register_early_pass($sess, false, box $name::new());
                )*}
            )
    }

    macro_rules! add_lint_group {
        ($sess:ident, $name:expr, $($lint:ident),*) => (
            store.register_group($sess, false, $name, vec![$(LintId::of($lint)),*]);
            )
    }

    add_early_builtin!(sess,
                       UnusedParens,
                       UnusedImportBraces,
                       AnonymousParameters,
                       UnusedDocComment,
                       );

    add_early_builtin_with_new!(sess,
                                DeprecatedAttr,
                                );

    add_builtin!(sess,
                 HardwiredLints,
                 WhileTrue,
                 ImproperCTypes,
                 VariantSizeDifferences,
                 BoxPointers,
                 UnusedAttributes,
                 PathStatements,
                 UnusedResults,
                 NonCamelCaseTypes,
                 NonSnakeCase,
                 NonUpperCaseGlobals,
                 NonShorthandFieldPatterns,
                 UnsafeCode,
                 UnusedAllocation,
                 MissingCopyImplementations,
                 UnstableFeatures,
                 UnconditionalRecursion,
                 InvalidNoMangleItems,
                 PluginAsLibrary,
                 MutableTransmutes,
                 UnionsWithDropFields,
                 UnreachablePub,
                 TypeAliasBounds,
                 UnusedBrokenConst,
                 );

    add_builtin_with_new!(sess,
                          TypeLimits,
                          MissingDoc,
                          MissingDebugImplementations,
                          );

    add_lint_group!(sess,
                    "bad_style",
                    NON_CAMEL_CASE_TYPES,
                    NON_SNAKE_CASE,
                    NON_UPPER_CASE_GLOBALS);

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
                    UNUSED_PARENS);

    add_lint_group!(sess,
                    "rust_2018_migration",
                    BARE_TRAIT_OBJECTS,
                    UNREACHABLE_PUB);

    // Guidelines for creating a future incompatibility lint:
    //
    // - Create a lint defaulting to warn as normal, with ideally the same error
    //   message you would normally give
    // - Add a suitable reference, typically an RFC or tracking issue. Go ahead
    //   and include the full URL, sort items in ascending order of issue numbers.
    // - Later, change lint to error
    // - Eventually, remove lint
    store.register_future_incompatible(sess,
                                       vec![
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
            id: LintId::of(LEGACY_IMPORTS),
            reference: "issue #38260 <https://github.com/rust-lang/rust/issues/38260>",
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
            edition: None,
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
            id: LintId::of(INCOHERENT_FUNDAMENTAL_IMPLS),
            reference: "issue #46205 <https://github.com/rust-lang/rust/issues/46205>",
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
            reference: "issue TBD",
            edition: Some(Edition::Edition2018),
        },
        ]);

    // Register renamed and removed lints
    store.register_renamed("single_use_lifetime", "single_use_lifetimes");
    store.register_renamed("elided_lifetime_in_path", "elided_lifetimes_in_paths");
    store.register_renamed("bare_trait_object", "bare_trait_objects");
    store.register_renamed("unstable_name_collision", "unstable_name_collisions");
    store.register_renamed("unused_doc_comment", "unused_doc_comments");
    store.register_renamed("unknown_features", "unused_features");
    store.register_removed("unsigned_negation", "replaced by negate_unsigned feature gate");
    store.register_removed("negate_unsigned", "cast a signed value instead");
    store.register_removed("raw_pointer_derive", "using derive with raw pointers is ok");
    // This was renamed to raw_pointer_derive, which was then removed,
    // so it is also considered removed
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
    store.register_removed("coerce_never",
        "converted into hard error, see https://github.com/rust-lang/rust/issues/48950");
    store.register_removed("resolve_trait_on_defaulted_unit",
        "converted into hard error, see https://github.com/rust-lang/rust/issues/48950");
}
