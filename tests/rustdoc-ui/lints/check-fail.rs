// compile-flags: -Z unstable-options --check

#![feature(rustdoc_missing_doc_code_examples)]
#![deny(missing_docs)]
#![deny(rustdoc::missing_doc_code_examples)]
#![deny(rustdoc::all)]

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
