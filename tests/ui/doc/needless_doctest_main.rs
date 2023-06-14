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
