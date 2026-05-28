//! Operators used to turn types into unsafe binders and back.

/// Unwrap an unsafe binder into its underlying type.
#[allow_internal_unstable(builtin_syntax)]
#[unstable(feature = "unsafe_binders", issue = "130516")]
pub macro unwrap_binder {
    ($expr:expr) => {
        builtin # unwrap_binder ( $expr )
    },
    ($expr:expr ; $ty:ty) => {
        builtin # unwrap_binder ( $expr, $ty )
    },
}

/// Wrap a type into an unsafe binder.
#[allow_internal_unstable(builtin_syntax)]
#[unstable(feature = "unsafe_binders", issue = "130516")]
pub macro wrap_binder {
    ($expr:expr) => {
        builtin # wrap_binder ( $expr )
    },
    ($expr:expr ; $ty:ty) => {
        builtin # wrap_binder ( $expr, $ty )
    },
}
