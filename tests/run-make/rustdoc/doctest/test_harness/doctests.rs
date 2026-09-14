// Test that we can successfully run two separate test suites.
// Check that we run all four tests even though `ill` and `bad` both fail.

//! ```test_harness
//! #[test]
//! fn well() {
//!     assert!(true);
//! }
//!
//! #[test]
//! fn ill() {
//!      assert!(false);
//! }
//! ```

//! ```test_harness
//! #[test]
//! fn bad() {
//!     assert!(false);
//! }
//!
//! #[test]
//! fn good() {
//!     assert!(true);
//! }
//! ```
