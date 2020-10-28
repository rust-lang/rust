macro_rules! declare_deprecated_lint {
    (pub $name: ident, $_reason: expr) => {
        declare_lint!(pub $name, Allow, "deprecated lint")
    }
}

declare_deprecated_lint! {
    /// **What it does:** Nothing. This lint has been deprecated.
    ///
    /// **Deprecation reason:** This used to check for `assert!(a == b)` and recommend
    /// replacement with `assert_eq!(a, b)`, but this is no longer needed after RFC 2011.
    pub SHOULD_ASSERT_EQ,
    "`assert!()` will be more flexible with RFC 2011"
}

declare_deprecated_lint! {
    /// **What it does:** Nothing. This lint has been deprecated.
    ///
    /// **Deprecation reason:** This used to check for `Vec::extend`, which was slower than
    /// `Vec::extend_from_slice`. Thanks to specialization, this is no longer true.
    pub EXTEND_FROM_SLICE,
    "`.extend_from_slice(_)` is a faster way to extend a Vec by a slice"
}

declare_deprecated_lint! {
    /// **What it does:** Nothing. This lint has been deprecated.
    ///
    /// **Deprecation reason:** `Range::step_by(0)` used to be linted since it's
    /// an infinite iterator, which is better expressed by `iter::repeat`,
    /// but the method has been removed for `Iterator::step_by` which panics
    /// if given a zero
    pub RANGE_STEP_BY_ZERO,
    "`iterator.step_by(0)` panics nowadays"
}

declare_deprecated_lint! {
    /// **What it does:** Nothing. This lint has been deprecated.
    ///
    /// **Deprecation reason:** This used to check for `Vec::as_slice`, which was unstable with good
    /// stable alternatives. `Vec::as_slice` has now been stabilized.
    pub UNSTABLE_AS_SLICE,
    "`Vec::as_slice` has been stabilized in 1.7"
}

declare_deprecated_lint! {
    /// **What it does:** Nothing. This lint has been deprecated.
    ///
    /// **Deprecation reason:** This used to check for `Vec::as_mut_slice`, which was unstable with good
    /// stable alternatives. `Vec::as_mut_slice` has now been stabilized.
    pub UNSTABLE_AS_MUT_SLICE,
    "`Vec::as_mut_slice` has been stabilized in 1.7"
}

declare_deprecated_lint! {
    /// **What it does:** Nothing. This lint has been deprecated.
    ///
    /// **Deprecation reason:** This used to check for `.to_string()` method calls on values
    /// of type `&str`. This is not unidiomatic and with specialization coming, `to_string` could be
    /// specialized to be as efficient as `to_owned`.
    pub STR_TO_STRING,
    "using `str::to_string` is common even today and specialization will likely happen soon"
}

declare_deprecated_lint! {
    /// **What it does:** Nothing. This lint has been deprecated.
    ///
    /// **Deprecation reason:** This used to check for `.to_string()` method calls on values
    /// of type `String`. This is not unidiomatic and with specialization coming, `to_string` could be
    /// specialized to be as efficient as `clone`.
    pub STRING_TO_STRING,
    "using `string::to_string` is common even today and specialization will likely happen soon"
}

declare_deprecated_lint! {
    /// **What it does:** Nothing. This lint has been deprecated.
    ///
    /// **Deprecation reason:** This lint should never have applied to non-pointer types, as transmuting
    /// between non-pointer types of differing alignment is well-defined behavior (it's semantically
    /// equivalent to a memcpy). This lint has thus been refactored into two separate lints:
    /// cast_ptr_alignment and transmute_ptr_to_ptr.
    pub MISALIGNED_TRANSMUTE,
    "this lint has been split into cast_ptr_alignment and transmute_ptr_to_ptr"
}

declare_deprecated_lint! {
    /// **What it does:** Nothing. This lint has been deprecated.
    ///
    /// **Deprecation reason:** This lint is too subjective, not having a good reason for being in clippy.
    /// Additionally, compound assignment operators may be overloaded separately from their non-assigning
    /// counterparts, so this lint may suggest a change in behavior or the code may not compile.
    pub ASSIGN_OPS,
    "using compound assignment operators (e.g., `+=`) is harmless"
}

declare_deprecated_lint! {
    /// **What it does:** Nothing. This lint has been deprecated.
    ///
    /// **Deprecation reason:** The original rule will only lint for `if let`. After
    /// making it support to lint `match`, naming as `if let` is not suitable for it.
    /// So, this lint is deprecated.
    pub IF_LET_REDUNDANT_PATTERN_MATCHING,
    "this lint has been changed to redundant_pattern_matching"
}

declare_deprecated_lint! {
    /// **What it does:** Nothing. This lint has been deprecated.
    ///
    /// **Deprecation reason:** This lint used to suggest replacing `let mut vec =
    /// Vec::with_capacity(n); vec.set_len(n);` with `let vec = vec![0; n];`. The
    /// replacement has very different performance characteristics so the lint is
    /// deprecated.
    pub UNSAFE_VECTOR_INITIALIZATION,
    "the replacement suggested by this lint had substantially different behavior"
}

declare_deprecated_lint! {
    /// **What it does:** Nothing. This lint has been deprecated.
    ///
    /// **Deprecation reason:** This lint has been superseded by the warn-by-default
    /// `invalid_value` rustc lint.
    pub INVALID_REF,
    "superseded by rustc lint `invalid_value`"
}

declare_deprecated_lint! {
    /// **What it does:** Nothing. This lint has been deprecated.
    ///
    /// **Deprecation reason:** This lint has been superseded by #[must_use] in rustc.
    pub UNUSED_COLLECT,
    "`collect` has been marked as #[must_use] in rustc and that covers all cases of this lint"
}

declare_deprecated_lint! {
    /// **What it does:** Nothing. This lint has been deprecated.
    ///
    /// **Deprecation reason:** This lint has been uplifted to rustc and is now called
    /// `array_into_iter`.
    pub INTO_ITER_ON_ARRAY,
    "this lint has been uplifted to rustc and is now called `array_into_iter`"
}

declare_deprecated_lint! {
    /// **What it does:** Nothing. This lint has been deprecated.
    ///
    /// **Deprecation reason:** This lint has been uplifted to rustc and is now called
    /// `unused_labels`.
    pub UNUSED_LABEL,
    "this lint has been uplifted to rustc and is now called `unused_labels`"
}

declare_deprecated_lint! {
    /// **What it does:** Nothing. This lint has been deprecated.
    ///
    /// **Deprecation reason:** Associated-constants are now preferred.
    pub REPLACE_CONSTS,
    "associated-constants `MIN`/`MAX` of integers are preferred to `{min,max}_value()` and module constants"
}

declare_deprecated_lint! {
    /// **What it does:** Nothing. This lint has been deprecated.
    ///
    /// **Deprecation reason:** The regex! macro does not exist anymore.
    pub REGEX_MACRO,
    "the regex! macro has been removed from the regex crate in 2018"
}

declare_deprecated_lint! {
    /// **What it does:** Nothing. This lint has been deprecated.
    ///
    /// **Deprecation reason:** This lint has been uplifted to rustc and is now called
    /// `drop_bounds`.
    pub DROP_BOUNDS,
    "this lint has been uplifted to rustc and is now called `drop_bounds`"
}

declare_deprecated_lint! {
    /// **What it does:** Nothing. This lint has been deprecated.
    ///
    /// **Deprecation reason:** This lint has been uplifted to rustc and is now called
    /// `temporary_cstring_as_ptr`.
    pub TEMPORARY_CSTRING_AS_PTR,
    "this lint has been uplifted to rustc and is now called `temporary_cstring_as_ptr`"
}
