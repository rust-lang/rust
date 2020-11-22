// compile-flags:--test
// edition:2018

//! ```
//! fn foo() {}
//!
//! mod bar {
//!     use super::foo;
//!     fn bar() {
//!         foo()
//!     }
//! }
//! ```
