//! Some lints that are built in to the compiler.
//!
//! These are the built-in lints that are emitted direct in the main
//! compiler code, rather than using their own custom pass. Those
//! lints are all available in `rustc_lint::builtin`.

use crate::lint::FutureIncompatibleInfo;
use crate::{declare_lint, declare_lint_pass};
use rustc_span::edition::Edition;
use rustc_span::symbol::sym;

declare_lint! {
    pub ILL_FORMED_ATTRIBUTE_INPUT,
    Deny,
    "ill-formed attribute inputs that were previously accepted and used in practice",
    @future_incompatible = FutureIncompatibleInfo {
        reference: "issue #57571 <https://github.com/rust-lang/rust/issues/57571>",
        edition: None,
    };
    crate_level_only
}

declare_lint! {
    pub CONFLICTING_REPR_HINTS,
    Deny,
    "conflicts between `#[repr(..)]` hints that were previously accepted and used in practice",
    @future_incompatible = FutureIncompatibleInfo {
        reference: "issue #68585 <https://github.com/rust-lang/rust/issues/68585>",
        edition: None,
    };
}

declare_lint! {
    pub META_VARIABLE_MISUSE,
    Allow,
    "possible meta-variable misuse at macro definition"
}

declare_lint! {
    pub INCOMPLETE_INCLUDE,
    Deny,
    "trailing content in included file"
}

declare_lint! {
    pub ARITHMETIC_OVERFLOW,
    Deny,
    "arithmetic operation overflows"
}

declare_lint! {
    pub UNCONDITIONAL_PANIC,
    Deny,
    "operation will cause a panic at runtime"
}

declare_lint! {
    pub CONST_ERR,
    Deny,
    "constant evaluation detected erroneous expression",
    report_in_external_macro
}

declare_lint! {
    pub UNUSED_IMPORTS,
    Warn,
    "imports that are never used"
}

declare_lint! {
    pub UNUSED_EXTERN_CRATES,
    Allow,
    "extern crates that are never used"
}

declare_lint! {
    pub UNUSED_CRATE_DEPENDENCIES,
    Allow,
    "crate dependencies that are never used",
    crate_level_only
}

declare_lint! {
    pub UNUSED_QUALIFICATIONS,
    Allow,
    "detects unnecessarily qualified names"
}

declare_lint! {
    pub UNKNOWN_LINTS,
    Warn,
    "unrecognized lint attribute"
}

declare_lint! {
    pub UNUSED_VARIABLES,
    Warn,
    "detect variables which are not used in any way"
}

declare_lint! {
    pub UNUSED_ASSIGNMENTS,
    Warn,
    "detect assignments that will never be read"
}

declare_lint! {
    pub DEAD_CODE,
    Warn,
    "detect unused, unexported items"
}

declare_lint! {
    pub UNUSED_ATTRIBUTES,
    Warn,
    "detects attributes that were not used by the compiler"
}

declare_lint! {
    pub UNREACHABLE_CODE,
    Warn,
    "detects unreachable code paths",
    report_in_external_macro
}

declare_lint! {
    pub UNREACHABLE_PATTERNS,
    Warn,
    "detects unreachable patterns"
}

declare_lint! {
    pub OVERLAPPING_PATTERNS,
    Warn,
    "detects overlapping patterns"
}

declare_lint! {
    pub BINDINGS_WITH_VARIANT_NAME,
    Warn,
    "detects pattern bindings with the same name as one of the matched variants"
}

declare_lint! {
    pub UNUSED_MACROS,
    Warn,
    "detects macros that were not used"
}

declare_lint! {
    pub WARNINGS,
    Warn,
    "mass-change the level for lints which produce warnings"
}

declare_lint! {
    pub UNUSED_FEATURES,
    Warn,
    "unused features found in crate-level `#[feature]` directives"
}

declare_lint! {
    pub STABLE_FEATURES,
    Warn,
    "stable features found in `#[feature]` directive"
}

declare_lint! {
    pub UNKNOWN_CRATE_TYPES,
    Deny,
    "unknown crate type found in `#[crate_type]` directive",
    crate_level_only
}

declare_lint! {
    pub TRIVIAL_CASTS,
    Allow,
    "detects trivial casts which could be removed"
}

declare_lint! {
    pub TRIVIAL_NUMERIC_CASTS,
    Allow,
    "detects trivial casts of numeric types which could be removed"
}

declare_lint! {
    pub PRIVATE_IN_PUBLIC,
    Warn,
    "detect private items in public interfaces not caught by the old implementation",
    @future_incompatible = FutureIncompatibleInfo {
        reference: "issue #34537 <https://github.com/rust-lang/rust/issues/34537>",
        edition: None,
    };
}

declare_lint! {
    pub EXPORTED_PRIVATE_DEPENDENCIES,
    Warn,
    "public interface leaks type from a private dependency"
}

declare_lint! {
    pub PUB_USE_OF_PRIVATE_EXTERN_CRATE,
    Deny,
    "detect public re-exports of private extern crates",
    @future_incompatible = FutureIncompatibleInfo {
        reference: "issue #34537 <https://github.com/rust-lang/rust/issues/34537>",
        edition: None,
    };
}

declare_lint! {
    pub INVALID_TYPE_PARAM_DEFAULT,
    Deny,
    "type parameter default erroneously allowed in invalid location",
    @future_incompatible = FutureIncompatibleInfo {
        reference: "issue #36887 <https://github.com/rust-lang/rust/issues/36887>",
        edition: None,
    };
}

declare_lint! {
    pub RENAMED_AND_REMOVED_LINTS,
    Warn,
    "lints that have been renamed or removed"
}

declare_lint! {
    pub UNALIGNED_REFERENCES,
    Allow,
    "detects unaligned references to fields of packed structs",
}

declare_lint! {
    pub SAFE_PACKED_BORROWS,
    Warn,
    "safe borrows of fields of packed structs were erroneously allowed",
    @future_incompatible = FutureIncompatibleInfo {
        reference: "issue #46043 <https://github.com/rust-lang/rust/issues/46043>",
        edition: None,
    };
}

declare_lint! {
    pub PATTERNS_IN_FNS_WITHOUT_BODY,
    Deny,
    "patterns in functions without body were erroneously allowed",
    @future_incompatible = FutureIncompatibleInfo {
        reference: "issue #35203 <https://github.com/rust-lang/rust/issues/35203>",
        edition: None,
    };
}

declare_lint! {
    pub LATE_BOUND_LIFETIME_ARGUMENTS,
    Warn,
    "detects generic lifetime arguments in path segments with late bound lifetime parameters",
    @future_incompatible = FutureIncompatibleInfo {
        reference: "issue #42868 <https://github.com/rust-lang/rust/issues/42868>",
        edition: None,
    };
}

declare_lint! {
    pub ORDER_DEPENDENT_TRAIT_OBJECTS,
    Deny,
    "trait-object types were treated as different depending on marker-trait order",
    @future_incompatible = FutureIncompatibleInfo {
        reference: "issue #56484 <https://github.com/rust-lang/rust/issues/56484>",
        edition: None,
    };
}

declare_lint! {
    pub COHERENCE_LEAK_CHECK,
    Warn,
    "distinct impls distinguished only by the leak-check code",
    @future_incompatible = FutureIncompatibleInfo {
        reference: "issue #56105 <https://github.com/rust-lang/rust/issues/56105>",
        edition: None,
    };
}

declare_lint! {
    pub DEPRECATED,
    Warn,
    "detects use of deprecated items",
    report_in_external_macro
}

declare_lint! {
    pub UNUSED_UNSAFE,
    Warn,
    "unnecessary use of an `unsafe` block"
}

declare_lint! {
    pub UNUSED_MUT,
    Warn,
    "detect mut variables which don't need to be mutable"
}

declare_lint! {
    pub UNCONDITIONAL_RECURSION,
    Warn,
    "functions that cannot return without calling themselves"
}

declare_lint! {
    pub SINGLE_USE_LIFETIMES,
    Allow,
    "detects lifetime parameters that are only used once"
}

declare_lint! {
    pub UNUSED_LIFETIMES,
    Allow,
    "detects lifetime parameters that are never used"
}

declare_lint! {
    pub TYVAR_BEHIND_RAW_POINTER,
    Warn,
    "raw pointer to an inference variable",
    @future_incompatible = FutureIncompatibleInfo {
        reference: "issue #46906 <https://github.com/rust-lang/rust/issues/46906>",
        edition: Some(Edition::Edition2018),
    };
}

declare_lint! {
    pub ELIDED_LIFETIMES_IN_PATHS,
    Allow,
    "hidden lifetime parameters in types are deprecated",
    crate_level_only
}

declare_lint! {
    pub BARE_TRAIT_OBJECTS,
    Warn,
    "suggest using `dyn Trait` for trait objects"
}

declare_lint! {
    pub ABSOLUTE_PATHS_NOT_STARTING_WITH_CRATE,
    Allow,
    "fully qualified paths that start with a module name \
     instead of `crate`, `self`, or an extern crate name",
     @future_incompatible = FutureIncompatibleInfo {
        reference: "issue #53130 <https://github.com/rust-lang/rust/issues/53130>",
        edition: Some(Edition::Edition2018),
     };
}

declare_lint! {
    pub ILLEGAL_FLOATING_POINT_LITERAL_PATTERN,
    Warn,
    "floating-point literals cannot be used in patterns",
    @future_incompatible = FutureIncompatibleInfo {
        reference: "issue #41620 <https://github.com/rust-lang/rust/issues/41620>",
        edition: None,
    };
}

declare_lint! {
    pub UNSTABLE_NAME_COLLISIONS,
    Warn,
    "detects name collision with an existing but unstable method",
    @future_incompatible = FutureIncompatibleInfo {
        reference: "issue #48919 <https://github.com/rust-lang/rust/issues/48919>",
        edition: None,
        // Note: this item represents future incompatibility of all unstable functions in the
        //       standard library, and thus should never be removed or changed to an error.
    };
}

declare_lint! {
    pub IRREFUTABLE_LET_PATTERNS,
    Warn,
    "detects irrefutable patterns in if-let and while-let statements"
}

declare_lint! {
    pub UNUSED_LABELS,
    Warn,
    "detects labels that are never used"
}

declare_lint! {
    pub BROKEN_INTRA_DOC_LINKS,
    Warn,
    "failures in resolving intra-doc link targets"
}

declare_lint! {
    pub INVALID_CODEBLOCK_ATTRIBUTES,
    Warn,
    "codeblock attribute looks a lot like a known one"
}

declare_lint! {
    pub MISSING_CRATE_LEVEL_DOCS,
    Allow,
    "detects crates with no crate-level documentation"
}

declare_lint! {
    pub MISSING_DOC_CODE_EXAMPLES,
    Allow,
    "detects publicly-exported items without code samples in their documentation"
}

declare_lint! {
    pub PRIVATE_DOC_TESTS,
    Allow,
    "detects code samples in docs of private items not documented by rustdoc"
}

declare_lint! {
    pub WHERE_CLAUSES_OBJECT_SAFETY,
    Warn,
    "checks the object safety of where clauses",
    @future_incompatible = FutureIncompatibleInfo {
        reference: "issue #51443 <https://github.com/rust-lang/rust/issues/51443>",
        edition: None,
    };
}

declare_lint! {
    pub PROC_MACRO_DERIVE_RESOLUTION_FALLBACK,
    Warn,
    "detects proc macro derives using inaccessible names from parent modules",
    @future_incompatible = FutureIncompatibleInfo {
        reference: "issue #50504 <https://github.com/rust-lang/rust/issues/50504>",
        edition: None,
    };
}

declare_lint! {
    pub MACRO_USE_EXTERN_CRATE,
    Allow,
    "the `#[macro_use]` attribute is now deprecated in favor of using macros \
     via the module system"
}

declare_lint! {
    pub MACRO_EXPANDED_MACRO_EXPORTS_ACCESSED_BY_ABSOLUTE_PATHS,
    Deny,
    "macro-expanded `macro_export` macros from the current crate \
     cannot be referred to by absolute paths",
    @future_incompatible = FutureIncompatibleInfo {
        reference: "issue #52234 <https://github.com/rust-lang/rust/issues/52234>",
        edition: None,
    };
    crate_level_only
}

declare_lint! {
    pub EXPLICIT_OUTLIVES_REQUIREMENTS,
    Allow,
    "outlives requirements can be inferred"
}

declare_lint! {
    pub INDIRECT_STRUCTURAL_MATCH,
    // defaulting to allow until rust-lang/rust#62614 is fixed.
    Allow,
    "pattern with const indirectly referencing non-structural-match type",
    @future_incompatible = FutureIncompatibleInfo {
        reference: "issue #62411 <https://github.com/rust-lang/rust/issues/62411>",
        edition: None,
    };
}

declare_lint! {
    pub DEPRECATED_IN_FUTURE,
    Allow,
    "detects use of items that will be deprecated in a future version",
    report_in_external_macro
}

declare_lint! {
    pub AMBIGUOUS_ASSOCIATED_ITEMS,
    Deny,
    "ambiguous associated items",
    @future_incompatible = FutureIncompatibleInfo {
        reference: "issue #57644 <https://github.com/rust-lang/rust/issues/57644>",
        edition: None,
    };
}

declare_lint! {
    pub MUTABLE_BORROW_RESERVATION_CONFLICT,
    Warn,
    "reservation of a two-phased borrow conflicts with other shared borrows",
    @future_incompatible = FutureIncompatibleInfo {
        reference: "issue #59159 <https://github.com/rust-lang/rust/issues/59159>",
        edition: None,
    };
}

declare_lint! {
    pub SOFT_UNSTABLE,
    Deny,
    "a feature gate that doesn't break dependent crates",
    @future_incompatible = FutureIncompatibleInfo {
        reference: "issue #64266 <https://github.com/rust-lang/rust/issues/64266>",
        edition: None,
    };
}

declare_lint! {
    pub INLINE_NO_SANITIZE,
    Warn,
    "detects incompatible use of `#[inline(always)]` and `#[no_sanitize(...)]`",
}

declare_lint! {
    pub ASM_SUB_REGISTER,
    Warn,
    "using only a subset of a register for inline asm inputs",
}

declare_lint! {
    pub UNSAFE_OP_IN_UNSAFE_FN,
    Allow,
    "unsafe operations in unsafe functions without an explicit unsafe block are deprecated",
    @feature_gate = sym::unsafe_block_in_unsafe_fn;
}

declare_lint! {
    pub CENUM_IMPL_DROP_CAST,
    Warn,
    "a C-like enum implementing Drop is cast",
    @future_incompatible = FutureIncompatibleInfo {
        reference: "issue #73333 <https://github.com/rust-lang/rust/issues/73333>",
        edition: None,
    };
}

declare_lint_pass! {
    /// Does nothing as a lint pass, but registers some `Lint`s
    /// that are used by other parts of the compiler.
    HardwiredLints => [
        ILLEGAL_FLOATING_POINT_LITERAL_PATTERN,
        ARITHMETIC_OVERFLOW,
        UNCONDITIONAL_PANIC,
        UNUSED_IMPORTS,
        UNUSED_EXTERN_CRATES,
        UNUSED_CRATE_DEPENDENCIES,
        UNUSED_QUALIFICATIONS,
        UNKNOWN_LINTS,
        UNUSED_VARIABLES,
        UNUSED_ASSIGNMENTS,
        DEAD_CODE,
        UNREACHABLE_CODE,
        UNREACHABLE_PATTERNS,
        OVERLAPPING_PATTERNS,
        BINDINGS_WITH_VARIANT_NAME,
        UNUSED_MACROS,
        WARNINGS,
        UNUSED_FEATURES,
        STABLE_FEATURES,
        UNKNOWN_CRATE_TYPES,
        TRIVIAL_CASTS,
        TRIVIAL_NUMERIC_CASTS,
        PRIVATE_IN_PUBLIC,
        EXPORTED_PRIVATE_DEPENDENCIES,
        PUB_USE_OF_PRIVATE_EXTERN_CRATE,
        INVALID_TYPE_PARAM_DEFAULT,
        CONST_ERR,
        RENAMED_AND_REMOVED_LINTS,
        UNALIGNED_REFERENCES,
        SAFE_PACKED_BORROWS,
        PATTERNS_IN_FNS_WITHOUT_BODY,
        LATE_BOUND_LIFETIME_ARGUMENTS,
        ORDER_DEPENDENT_TRAIT_OBJECTS,
        COHERENCE_LEAK_CHECK,
        DEPRECATED,
        UNUSED_UNSAFE,
        UNUSED_MUT,
        UNCONDITIONAL_RECURSION,
        SINGLE_USE_LIFETIMES,
        UNUSED_LIFETIMES,
        UNUSED_LABELS,
        TYVAR_BEHIND_RAW_POINTER,
        ELIDED_LIFETIMES_IN_PATHS,
        BARE_TRAIT_OBJECTS,
        ABSOLUTE_PATHS_NOT_STARTING_WITH_CRATE,
        UNSTABLE_NAME_COLLISIONS,
        IRREFUTABLE_LET_PATTERNS,
        BROKEN_INTRA_DOC_LINKS,
        INVALID_CODEBLOCK_ATTRIBUTES,
        MISSING_CRATE_LEVEL_DOCS,
        MISSING_DOC_CODE_EXAMPLES,
        PRIVATE_DOC_TESTS,
        WHERE_CLAUSES_OBJECT_SAFETY,
        PROC_MACRO_DERIVE_RESOLUTION_FALLBACK,
        MACRO_USE_EXTERN_CRATE,
        MACRO_EXPANDED_MACRO_EXPORTS_ACCESSED_BY_ABSOLUTE_PATHS,
        ILL_FORMED_ATTRIBUTE_INPUT,
        CONFLICTING_REPR_HINTS,
        META_VARIABLE_MISUSE,
        DEPRECATED_IN_FUTURE,
        AMBIGUOUS_ASSOCIATED_ITEMS,
        MUTABLE_BORROW_RESERVATION_CONFLICT,
        INDIRECT_STRUCTURAL_MATCH,
        SOFT_UNSTABLE,
        INLINE_NO_SANITIZE,
        ASM_SUB_REGISTER,
        UNSAFE_OP_IN_UNSAFE_FN,
        INCOMPLETE_INCLUDE,
        CENUM_IMPL_DROP_CAST,
    ]
}

declare_lint! {
    pub UNUSED_DOC_COMMENTS,
    Warn,
    "detects doc comments that aren't used by rustdoc"
}

declare_lint_pass!(UnusedDocComment => [UNUSED_DOC_COMMENTS]);
