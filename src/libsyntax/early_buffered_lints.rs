//! Allows the buffering of lints for later.
//!
//! Since we cannot have a dependency on `librustc`, we implement some types here that are somewhat
//! redundant. Later, these types can be converted to types for use by the rest of the compiler.

use rustc_session::lint::FutureIncompatibleInfo;
use rustc_session::declare_lint;
pub use rustc_session::lint::BufferedEarlyLint;

declare_lint! {
    pub ILL_FORMED_ATTRIBUTE_INPUT,
    Deny,
    "ill-formed attribute inputs that were previously accepted and used in practice",
    @future_incompatible = FutureIncompatibleInfo {
        reference: "issue #57571 <https://github.com/rust-lang/rust/issues/57571>",
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
