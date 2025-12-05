#![warn(clippy::needless_doctest_main)]

/// Description
/// ```rust
/// fn main() {
//~^ needless_doctest_main
///     let a = 0;
/// }
/// ```
fn mulpipulpi() {}

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

/// Ignore code block, should not lint at all
/// ```rust, ignore
/// fn main() {
///     // Hi!
///     let _ = 0;
/// }
/// ```
fn mpulpilumi() {}

/// Spaces in weird positions (including an \u{A0} after `main`)
/// ```rust
/// fn     main (){
//~^ needless_doctest_main
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
