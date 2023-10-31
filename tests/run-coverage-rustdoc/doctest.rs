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
//! #[derive(Debug, PartialEq)]
//! struct SomeError {
//!     msg: String,
//! }
//! let mut res = Err(SomeError { msg: String::from("a message") });
//! if res.is_ok() {
//!     res?;
//! } else {
//!     if *res.as_ref().unwrap_err() == *res.as_ref().unwrap_err() {
//!         println!("{:?}", res);
//!     }
//!     if *res.as_ref().unwrap_err() == *res.as_ref().unwrap_err() {
//!         res = Ok(1);
//!     }
//!     res = Ok(0);
//! }
//! // need to be explicit because rustdoc cant infer the return type
//! Ok::<(), SomeError>(())
//! ```
//!
//! doctest with custom main:
//! ```
//! fn some_func() {
//!     println!("called some_func()");
//! }
//!
//! #[derive(Debug)]
//! struct SomeError;
//!
//! extern crate doctest_crate;
//!
//! fn doctest_main() -> Result<(), SomeError> {
//!     some_func();
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
// aux-build:doctest_crate.rs
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

// FIXME(Swatinem): Fix known issue that coverage code region columns need to be offset by the
// doc comment line prefix (`///` or `//!`) and any additional indent (before or after the doc
// comment characters). This test produces `llvm-cov show` results demonstrating the problem.
//
// One of the above tests now includes: `derive(Debug, PartialEq)`, producing an `llvm-cov show`
// result with a distinct count for `Debug`, denoted by `^1`, but the caret points to the wrong
// column. Similarly, the `if` blocks without `else` blocks show `^0`, which should point at, or
// one character past, the `if` block's closing brace. In both cases, these are most likely off
// by the number of characters stripped from the beginning of each doc comment line: indent
// whitespace, if any, doc comment prefix (`//!` in this case) and (I assume) one space character
// (?). Note, when viewing `llvm-cov show` results in `--color` mode, the column offset errors are
// more pronounced, and show up in more places, with background color used to show some distinct
// code regions with different coverage counts.
//
// NOTE: Since the doc comment line prefix may vary, one possible solution is to replace each
// character stripped from the beginning of doc comment lines with a space. This will give coverage
// results the correct column offsets, and I think it should compile correctly, but I don't know
// what affect it might have on diagnostic messages from the compiler, and whether anyone would care
// if the indentation changed. I don't know if there is a more viable solution.
