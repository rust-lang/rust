//@ compile-flags: --test --test-args=--test-threads=1
//@ normalize-stdout: "finished in \d+\.\d+s" -> "finished in $$TIME"
//@ failure-status: 101

//! ```
//! #[should_panic]
//! ```

//! ```
//! fn main() {
//!     #[should_panic]
//! }
//! ```

//! ```
//! fn main() { }
//! #[should_panic]
//! ```

//! ```
//! let x = 0; #[should_panic]
//! ```

//! ```
//! let x = 0; //! assert!(true);
//! ```

/// ```
/// /// doc comment
/// ```
struct Test;

/// ```
/// #[cfg(true)] {
/// }    /// ```
pub fn wtf() {}
