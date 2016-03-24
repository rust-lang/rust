macro_rules! declare_deprecated_lint {
    (pub $name: ident, $_reason: expr) => {
        declare_lint!(pub $name, Allow, "deprecated lint")
    }
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
