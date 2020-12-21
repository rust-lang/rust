//! This test ensures that code from doctests is properly re-mapped.
//! See <https://github.com/rust-lang/rust/issues/79417> for more info.
//!
//! Just some random code:
//! ```
//! if true {
//!     // this is executed!
//!     assert_eq!(1, 1);
//! } else {
//!     // this is not!
//!     assert_eq!(1, 2);
//! }
//! ```
//!
//! doctest testing external code:
//! ```
//! extern crate doctest_crate;
//! doctest_crate::fn_run_in_doctests(1);
//! ```
//!
//! doctest returning a result:
//! ```
//! #[derive(Debug)]
//! struct SomeError;
//! let mut res = Err(SomeError);
//! if res.is_ok() {
//!   res?;
//! } else {
//!   res = Ok(0);
//! }
//! // need to be explicit because rustdoc cant infer the return type
//! Ok::<(), SomeError>(())
//! ```
//!
//! doctest with custom main:
//! ```
//! #[derive(Debug)]
//! struct SomeError;
//!
//! extern crate doctest_crate;
//!
//! fn doctest_main() -> Result<(), SomeError> {
//!     doctest_crate::fn_run_in_doctests(2);
//!     Ok(())
//! }
//!
//! // this `main` is not shown as covered, as it clashes with all the other
//! // `main` functions that were automatically generated for doctests
//! fn main() -> Result<(), SomeError> {
//!     doctest_main()
//! }
//! ```

/// doctest attached to fn testing external code:
/// ```
/// extern crate doctest_crate;
/// doctest_crate::fn_run_in_doctests(3);
/// ```
///
fn main() {
    if true {
        assert_eq!(1, 1);
    } else {
        assert_eq!(1, 2);
    }
}
