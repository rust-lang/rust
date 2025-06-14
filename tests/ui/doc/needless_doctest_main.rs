//@ check-pass

#![warn(clippy::needless_doctest_main)]
//! issue 10491:
//! ```rust,no_test
//! use std::collections::HashMap;
//!
//! fn main() {
//!     let mut m = HashMap::new();
//!     m.insert(1u32, 2u32);
//! }
//! ```

/// some description here
/// ```rust,no_test
/// fn main() {
///     foo()
/// }
/// ```
fn foo() {}

fn main() {}

fn issue8244() -> Result<(), ()> {
    //! ```compile_fail
    //! fn test() -> Result< {}
    //! ```
    Ok(())
}

/// # Examples
///
/// ```
/// use std::error::Error;
/// fn main() -> Result<(), Box<dyn Error>/* > */ {
/// }
/// ```
fn issue15041() {}
