#![deny(clippy::internal)]
#![feature(rustc_private)]

#[macro_use]
extern crate clippy_lints;
use clippy_lints::deprecated_lints::ClippyDeprecatedLint;

declare_deprecated_lint! {
    /// ### What it does
    /// Nothing. This lint has been deprecated.
    ///
    /// ### Deprecation reason
    /// TODO
    #[clippy::version = "1.63.0"]
    pub COOL_LINT_DEFAULT,
    "default deprecation note"
}

declare_deprecated_lint! {
    /// ### What it does
    /// Nothing. This lint has been deprecated.
    ///
    /// ### Deprecation reason
    /// This lint has been replaced by `cooler_lint`
    #[clippy::version = "1.63.0"]
    pub COOL_LINT,
    "this lint has been replaced by `cooler_lint`"
}

fn main() {}
