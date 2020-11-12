// compile-flags: -Z unstable-options --check

#![deny(missing_docs)]
#![deny(rustdoc)]

//! ```rust,testharness
//~^ ERROR
//! let x = 12;
//! ```

pub fn foo() {}
//~^ ERROR
//~^^ ERROR

/// hello
//~^ ERROR
///
/// ```rust,testharness
/// let x = 12;
/// ```
pub fn bar() {}
