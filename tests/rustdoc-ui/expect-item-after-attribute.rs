//@ compile-flags: --test
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
