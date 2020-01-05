//! Allows the buffering of lints for later.
//!
//! Since we cannot have a dependency on `librustc`, we implement some types here that are somewhat
//! redundant. Later, these types can be converted to types for use by the rest of the compiler.

pub use rustc_session::lint::builtin::ILL_FORMED_ATTRIBUTE_INPUT;
pub use rustc_session::lint::builtin::{INCOMPLETE_INCLUDE, META_VARIABLE_MISUSE};
pub use rustc_session::lint::BufferedEarlyLint;
