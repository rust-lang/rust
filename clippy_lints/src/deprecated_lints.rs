macro_rules! declare_deprecated_lint {
    (pub $name: ident, $_reason: expr) => {
        declare_lint!(pub $name, Allow, "deprecated lint")
    }
}


/// **What it does:** Nothing. This lint has been deprecated.
///
/// **Deprecation reason:** This used to check for `Vec::extend`, which was slower than
/// `Vec::extend_from_slice`. Thanks to specialization, this is no longer true.
declare_deprecated_lint! {
    pub EXTEND_FROM_SLICE,
    "`.extend_from_slice(_)` is a faster way to extend a Vec by a slice"
}

/// **What it does:** Nothing. This lint has been deprecated.
///
/// **Deprecation reason:** `Range::step_by(0)` used to be linted since it's
/// an infinite iterator, which is better expressed by `iter::repeat`,
/// but the method has been removed for `Iterator::step_by` which panics
/// if given a zero
declare_deprecated_lint! {
    pub RANGE_STEP_BY_ZERO,
    "`iterator.step_by(0)` panics nowadays"
}

/// **What it does:** Nothing. This lint has been deprecated.
///
/// **Deprecation reason:** This used to check for `Vec::as_slice`, which was unstable with good
/// stable alternatives. `Vec::as_slice` has now been stabilized.
declare_deprecated_lint! {
    pub UNSTABLE_AS_SLICE,
    "`Vec::as_slice` has been stabilized in 1.7"
}


/// **What it does:** Nothing. This lint has been deprecated.
///
/// **Deprecation reason:** This used to check for `Vec::as_mut_slice`, which was unstable with good
/// stable alternatives. `Vec::as_mut_slice` has now been stabilized.
declare_deprecated_lint! {
    pub UNSTABLE_AS_MUT_SLICE,
    "`Vec::as_mut_slice` has been stabilized in 1.7"
}

/// **What it does:** Nothing. This lint has been deprecated.
///
/// **Deprecation reason:** This used to check for `.to_string()` method calls on values
/// of type `&str`. This is not unidiomatic and with specialization coming, `to_string` could be
/// specialized to be as efficient as `to_owned`.
declare_deprecated_lint! {
    pub STR_TO_STRING,
    "using `str::to_string` is common even today and specialization will likely happen soon"
}

/// **What it does:** Nothing. This lint has been deprecated.
///
/// **Deprecation reason:** This used to check for `.to_string()` method calls on values
/// of type `String`. This is not unidiomatic and with specialization coming, `to_string` could be
/// specialized to be as efficient as `clone`.
declare_deprecated_lint! {
    pub STRING_TO_STRING,
    "using `string::to_string` is common even today and specialization will likely happen soon"
}
