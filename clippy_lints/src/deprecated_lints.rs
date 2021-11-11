// NOTE: if you add a deprecated lint in this file, please add a corresponding test in
// tests/ui/deprecated.rs

/// This struct fakes the `Lint` declaration that is usually created by `declare_lint!`. This
/// enables the simple extraction of the metadata without changing the current deprecation
/// declaration.
pub struct ClippyDeprecatedLint;

macro_rules! declare_deprecated_lint {
    { $(#[$attr:meta])* pub $name: ident, $_reason: expr} => {
        $(#[$attr])*
        #[allow(dead_code)]
        pub static $name: ClippyDeprecatedLint = ClippyDeprecatedLint {};
    }
}

declare_deprecated_lint! {
    /// ### What it does
    /// Nothing. This lint has been deprecated.
    ///
    /// ### Deprecation reason
    /// This used to check for `assert!(a == b)` and recommend
    /// replacement with `assert_eq!(a, b)`, but this is no longer needed after RFC 2011.
    #[clippy::version = "pre 1.29.0"]
    pub SHOULD_ASSERT_EQ,
    "`assert!()` will be more flexible with RFC 2011"
}

declare_deprecated_lint! {
    /// ### What it does
    /// Nothing. This lint has been deprecated.
    ///
    /// ### Deprecation reason
    /// This used to check for `Vec::extend`, which was slower than
    /// `Vec::extend_from_slice`. Thanks to specialization, this is no longer true.
    #[clippy::version = "pre 1.29.0"]
    pub EXTEND_FROM_SLICE,
    "`.extend_from_slice(_)` is a faster way to extend a Vec by a slice"
}

declare_deprecated_lint! {
    /// ### What it does
    /// Nothing. This lint has been deprecated.
    ///
    /// ### Deprecation reason
    /// `Range::step_by(0)` used to be linted since it's
    /// an infinite iterator, which is better expressed by `iter::repeat`,
    /// but the method has been removed for `Iterator::step_by` which panics
    /// if given a zero
    #[clippy::version = "pre 1.29.0"]
    pub RANGE_STEP_BY_ZERO,
    "`iterator.step_by(0)` panics nowadays"
}

declare_deprecated_lint! {
    /// ### What it does
    /// Nothing. This lint has been deprecated.
    ///
    /// ### Deprecation reason
    /// This used to check for `Vec::as_slice`, which was unstable with good
    /// stable alternatives. `Vec::as_slice` has now been stabilized.
    #[clippy::version = "pre 1.29.0"]
    pub UNSTABLE_AS_SLICE,
    "`Vec::as_slice` has been stabilized in 1.7"
}

declare_deprecated_lint! {
    /// ### What it does
    /// Nothing. This lint has been deprecated.
    ///
    /// ### Deprecation reason
    /// This used to check for `Vec::as_mut_slice`, which was unstable with good
    /// stable alternatives. `Vec::as_mut_slice` has now been stabilized.
    #[clippy::version = "pre 1.29.0"]
    pub UNSTABLE_AS_MUT_SLICE,
    "`Vec::as_mut_slice` has been stabilized in 1.7"
}

declare_deprecated_lint! {
    /// ### What it does
    /// Nothing. This lint has been deprecated.
    ///
    /// ### Deprecation reason
    /// This lint should never have applied to non-pointer types, as transmuting
    /// between non-pointer types of differing alignment is well-defined behavior (it's semantically
    /// equivalent to a memcpy). This lint has thus been refactored into two separate lints:
    /// cast_ptr_alignment and transmute_ptr_to_ptr.
    #[clippy::version = "pre 1.29.0"]
    pub MISALIGNED_TRANSMUTE,
    "this lint has been split into cast_ptr_alignment and transmute_ptr_to_ptr"
}

declare_deprecated_lint! {
    /// ### What it does
    /// Nothing. This lint has been deprecated.
    ///
    /// ### Deprecation reason
    /// This lint is too subjective, not having a good reason for being in clippy.
    /// Additionally, compound assignment operators may be overloaded separately from their non-assigning
    /// counterparts, so this lint may suggest a change in behavior or the code may not compile.
    #[clippy::version = "1.30.0"]
    pub ASSIGN_OPS,
    "using compound assignment operators (e.g., `+=`) is harmless"
}

declare_deprecated_lint! {
    /// ### What it does
    /// Nothing. This lint has been deprecated.
    ///
    /// ### Deprecation reason
    /// The original rule will only lint for `if let`. After
    /// making it support to lint `match`, naming as `if let` is not suitable for it.
    /// So, this lint is deprecated.
    #[clippy::version = "pre 1.29.0"]
    pub IF_LET_REDUNDANT_PATTERN_MATCHING,
    "this lint has been changed to redundant_pattern_matching"
}

declare_deprecated_lint! {
    /// ### What it does
    /// Nothing. This lint has been deprecated.
    ///
    /// ### Deprecation reason
    /// This lint used to suggest replacing `let mut vec =
    /// Vec::with_capacity(n); vec.set_len(n);` with `let vec = vec![0; n];`. The
    /// replacement has very different performance characteristics so the lint is
    /// deprecated.
    #[clippy::version = "pre 1.29.0"]
    pub UNSAFE_VECTOR_INITIALIZATION,
    "the replacement suggested by this lint had substantially different behavior"
}

declare_deprecated_lint! {
    /// ### What it does
    /// Nothing. This lint has been deprecated.
    ///
    /// ### Deprecation reason
    /// This lint has been superseded by #[must_use] in rustc.
    #[clippy::version = "1.39.0"]
    pub UNUSED_COLLECT,
    "`collect` has been marked as #[must_use] in rustc and that covers all cases of this lint"
}

declare_deprecated_lint! {
    /// ### What it does
    /// Nothing. This lint has been deprecated.
    ///
    /// ### Deprecation reason
    /// Associated-constants are now preferred.
    #[clippy::version = "1.44.0"]
    pub REPLACE_CONSTS,
    "associated-constants `MIN`/`MAX` of integers are preferred to `{min,max}_value()` and module constants"
}

declare_deprecated_lint! {
    /// ### What it does
    /// Nothing. This lint has been deprecated.
    ///
    /// ### Deprecation reason
    /// The regex! macro does not exist anymore.
    #[clippy::version = "1.47.0"]
    pub REGEX_MACRO,
    "the regex! macro has been removed from the regex crate in 2018"
}

declare_deprecated_lint! {
    /// ### What it does
    /// Nothing. This lint has been deprecated.
    ///
    /// ### Deprecation reason
    /// This lint has been replaced by `manual_find_map`, a
    /// more specific lint.
    #[clippy::version = "1.51.0"]
    pub FIND_MAP,
    "this lint has been replaced by `manual_find_map`, a more specific lint"
}

declare_deprecated_lint! {
    /// ### What it does
    /// Nothing. This lint has been deprecated.
    ///
    /// ### Deprecation reason
    /// This lint has been replaced by `manual_filter_map`, a
    /// more specific lint.
    #[clippy::version = "1.53.0"]
    pub FILTER_MAP,
    "this lint has been replaced by `manual_filter_map`, a more specific lint"
}

declare_deprecated_lint! {
    /// ### What it does
    /// Nothing. This lint has been deprecated.
    ///
    /// ### Deprecation reason
    /// The `avoid_breaking_exported_api` config option was added, which
    /// enables the `enum_variant_names` lint for public items.
    /// ```
    #[clippy::version = "1.54.0"]
    pub PUB_ENUM_VARIANT_NAMES,
    "set the `avoid-breaking-exported-api` config option to `false` to enable the `enum_variant_names` lint for public items"
}

declare_deprecated_lint! {
    /// ### What it does
    /// Nothing. This lint has been deprecated.
    ///
    /// ### Deprecation reason
    /// The `avoid_breaking_exported_api` config option was added, which
    /// enables the `wrong_self_conversion` lint for public items.
    #[clippy::version = "1.54.0"]
    pub WRONG_PUB_SELF_CONVENTION,
    "set the `avoid-breaking-exported-api` config option to `false` to enable the `wrong_self_convention` lint for public items"
}
