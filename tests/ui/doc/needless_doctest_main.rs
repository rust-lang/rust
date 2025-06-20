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

#[rustfmt::skip]
/// Description
/// ```rust
/// fn main() {
//~^ error: needless `fn main` in doctest
///     let a = 0;
/// }
/// ```
fn mulpipulpi() {}

#[rustfmt::skip]
/// With a `#[no_main]`
/// ```rust
/// #[no_main]
/// fn a() {
///     let _ = 0;
/// }
/// ```
fn pulpimulpi() {}

// Without a `#[no_main]` attribute
/// ```rust
/// fn a() {
///     let _ = 0;
/// }
/// ```
fn plumilupi() {}

#[rustfmt::skip]
/// Additional function, shouldn't trigger
/// ```rust
/// fn additional_function() {
///     let _ = 0;
///     // Thus `fn main` is actually relevant!
/// }
/// fn main() {
///     let _ = 0;
/// }
/// ```
fn mlupipupi() {}

#[rustfmt::skip]
/// Additional function AFTER main, shouldn't trigger
/// ```rust
/// fn main() {
///     let _ = 0;
/// }
/// fn additional_function() {
///     let _ = 0;
///     // Thus `fn main` is actually relevant!
/// }
/// ```
fn lumpimupli() {}

#[rustfmt::skip]
/// Ignore code block, should not lint at all
/// ```rust, ignore
/// fn main() {
//~^ error: needless `fn main` in doctest
///     // Hi!
///     let _ = 0;
/// }
/// ```
fn mpulpilumi() {}

#[rustfmt::skip]
/// Spaces in weird positions (including an \u{A0} after `main`)
/// ```rust
/// fn     main (){
//~^ error: needless `fn main` in doctest
///     let _ = 0;
/// }
/// ```
fn plumpiplupi() {}

/// 4 Functions, this should not lint because there are several function
///
/// ```rust
/// fn a() {let _ = 0; }
/// fn b() {let _ = 0; }
/// fn main() { let _ = 0; }
/// fn d() { let _ = 0; }
/// ```
fn pulmipulmip() {}

/// 3 Functions but main is first, should also not lint
///
///```rust
/// fn main() { let _ = 0; }
/// fn b() { let _ = 0; }
/// fn c() { let _ = 0; }
/// ```
fn pmuplimulip() {}

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
