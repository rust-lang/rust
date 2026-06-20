#[doc(attribute = "must_use")]
//
/// Warn when a value is ignored.
///
/// The `must_use` attribute applies to values where simply creating or returning them is
/// often not enough. If a value marked with `#[must_use]` is produced and then ignored, the
/// compiler warns through the [`unused_must_use`] lint.
///
/// This is most common on types that represent an important state or outcome. For example,
/// [`Result`] is marked `#[must_use]` because ignoring an error value can hide a failed operation.
/// In the following example, the returned `Result` is the only sign that writing the message
/// might have failed:
///
/// ```rust
/// # #![allow(unused_must_use)]
/// fn write_message() -> std::io::Result<()> {
///     // Write the message...
///     Ok(())
/// }
///
/// write_message();
/// ```
///
/// Ignoring that `Result` triggers this warning:
///
/// ```text
/// warning: unused `Result` that must be used
///   = note: this `Result` may be an `Err` variant, which should be handled
///   = note: `#[warn(unused_must_use)]` (part of `#[warn(unused)]`) on by default
/// help: use `let _ = ...` to ignore the resulting value
/// ```
///
/// Future values are also `#[must_use]`: creating a future does not run it, so ignoring one often
/// means the intended asynchronous work never happens.
///
/// You can also place `#[must_use]` on a function, method, or trait declaration. On a function or
/// method, the warning is tied to ignoring that call's return value:
///
/// ```rust
/// # #![allow(unused_must_use)]
/// #[must_use]
/// fn make_token() -> String {
///     String::from("token")
/// }
///
/// // Ignoring this call's return value triggers `unused_must_use`.
/// make_token();
/// ```
///
/// On a trait, the warning applies when a function returns an opaque type (`impl Trait`) or trait
/// object (`dyn Trait`) whose bounds include that trait. This is how futures warn if you create one
/// but never poll or await it, since an `async fn` returns an opaque type implementing [`Future`].
///
/// The attribute can include a message explaining what the caller should do with the value:
///
/// ```rust
/// # #![allow(dead_code)]
/// #[must_use = "call `.finish()` to complete the operation"]
/// fn start_operation() -> Operation {
///     Operation
/// }
///
/// struct Operation;
/// ```
///
/// If intentionally ignoring the value is correct, bind it to `_` or call [`drop`]:
///
/// ```rust
/// # #[must_use]
/// # fn make_token() -> String {
/// #     String::from("token")
/// # }
/// let _ = make_token();
/// drop(make_token());
/// ```
///
/// The attribute is a warning tool, not a type-system rule. Code can still explicitly discard a
/// `#[must_use]` value, and the compiler does not require callers to inspect or otherwise act on
/// the value.
///
/// For more information, see the Reference on [the `must_use` attribute].
///
/// [`Result`]: result::Result
/// [`Future`]: future::Future
/// [`unused_must_use`]: ../rustc/lints/listing/warn-by-default.html#unused-must-use
/// [the `must_use` attribute]: ../reference/attributes/diagnostics.html#the-must_use-attribute
mod must_use_attribute {}

#[doc(attribute = "allow")]
//
/// The `allow` attribute suppresses lint checks that would otherwise produce
/// warnings or errors. It can be used on any lint or lint group (except those
/// set to `forbid`).
///
/// ```rust
/// #[allow(dead_code)]
/// fn unused_function() {
/// }
/// fn main() {
///   // unused_function does not generate a compiler warning.
/// }
/// ```
///
/// Without `#[allow(dead_code)]`, the example above would emit:
///
/// ```text
/// warning: function `unused_function` is never used
///  --> main.rs:1:4
///   |
/// 1 | fn unused_function() {
///   |    ^^^^^^^^^^^^^^^
///   |
///   = note: `#[warn(dead_code)]` (part of `#[warn(unused)]`) on by default
///
///   warning: 1 warning emitted
///
/// ```
///
/// Multiple warnings can be suppressed at once by separating names with commas:
///
/// ```rust
/// #[allow(unused_variables, unused_mut)]
/// fn main() {
///     let mut x: u32 = 42;
/// }
///
/// ```
///
/// To apply `allow` to an entire module or crate, use the inner attribute syntax
/// `#![allow(...)]` instead.
///
/// ```rust
/// #![allow(dead_code)]
/// fn unused_foo() {
/// }
/// fn unused_bar() {
/// }
/// fn main() {
/// }
///
/// ```
///
/// This is mostly used to prevent lint warnings or errors while still under development.
/// It's also important to consider that overusing `allow` could make code harder to maintain
/// and possibly hide issues. It cannot override a lint that has been set to forbid.
///
/// For more information, see the Reference on [the `allow` attribute].
///
/// [the `allow` attribute]: ../reference/attributes/diagnostics.html#lint-check-attributes
mod allow_attribute {}

#[doc(attribute = "cfg")]
//
/// Used for conditional compilation.
///
/// The `cfg` attribute includes or removes the code below it based on a condition,
/// like target OS, target architecture or custom flag. If the condition is true the code
/// stays (and the `cfg` attribute is removed). If it's false the code is not compiled at all.
///
/// ```rust
/// #[cfg(target_os = "linux")]
/// fn platform_specific() {
///     println!("Running on Linux");
/// }
///
/// #[cfg(not(target_os = "linux"))]
/// fn platform_specific() {
///     println!("Running on something else");
/// }
///
/// ```
///
/// Only one of these two functions is compiled depending on the target, while the other is removed
/// before compilation.
///
/// Common conditions include `target_os`, `target_arch`, `target_endian`, `unix`, `windows`,
/// `macos`, `test` (set when running tests), and custom flags set with `--cfg` or
/// Cargo features (`feature = "..."`).
///
/// Conditions can also be combined with `all(...)`, `any(...)`, `not(...)`.
///
/// ```rust
/// #[cfg(all(unix, target_pointer_width = "64"))]
/// fn unix_64bit() {
/// }
///
/// ```
///
/// You can also use `cfg` more than once on the same item. The item is only
/// kept if all the conditions are true, same as combining them with `all(...)`.
///
/// For a check you can use inside a function, see the [`cfg!`] macro. To
/// conditionally apply a different attribute, see [`cfg_attr`].
///
/// For more information, see the Reference on [the `cfg` attribute].
///
/// [`cfg_attr`]: ../reference/conditional-compilation.html#the-cfg_attr-attribute
/// [the `cfg` attribute]: ../reference/conditional-compilation.html#the-cfg-attribute
mod cfg_attribute {}

#[doc(attribute = "deny")]
//
/// Signals an error when a lint check is violated. This is useful for enforcing rules or
/// preventing certain patterns.
///
/// Unlike `allow`, which suppresses lints, `deny` makes them hard error instead of warnings, `deny` can be
/// overridden by `allow` and `warn` in inner scopes.
///
/// ```rust, compile_fail
/// #[deny(unused)]
/// fn foo() {
///     let x = 42;
/// }
///
/// fn main() {
///     foo(); // foo errors instead of warn because it's set to `#[deny(unused)]`
/// }
/// ```
///
/// Without `#[deny(unused)]`, the example above would only emit a warning.
///
/// ```rust
/// #![deny(dead_code)]
/// #[allow(dead_code)]
/// fn allowed_function() {} // No error `deny` was overridden by `allow`.
/// ```
///
/// For more information, see the Reference on [the `deny` attribute].
///
/// [the `deny` attribute]: ../reference/attributes/diagnostics.html#lint-check-attributes
mod deny_attribute {}

#[doc(attribute = "forbid")]
//
/// Turns a lint into a hard error and prevents it from being overridden.
///
/// A lint set to forbid cannot be overridden by allow or warn in any inner scope,
/// attempting either is a compile error. Writing #[deny(...)] on the same lint inside a
/// forbid scope is permitted, but has no effect; the lint remains at the forbid level.
///
/// This is useful for enforcing strict policies that should not be relaxed
/// anywhere in the codebase.
///
/// ```rust
/// #![forbid(unsafe_code)]
///
/// // This would cause a compilation error if uncommented:
/// // #[allow(unsafe_code)] // error: cannot override `forbid`
/// ```
///
/// Multiple lints can be set to `forbid` at once:
///
/// ```rust
/// #![forbid(unsafe_code, unused)]
/// ```
///
/// To apply `forbid` to an entire module or crate, use the inner attribute
/// syntax `#![forbid(...)]` at the crate root. To apply it to a smaller
/// scope, use `#[forbid(...)]` on the specific item.
///
/// The lint checks supported by rustc can be found via `rustc -W help`,
/// along with their default settings and are documented in the rustc book.
///
/// For more information, see the Reference on [the `forbid` attribute].
///
/// [the `forbid` attribute]: ../reference/attributes/diagnostics.html#lint-check-attributes
mod forbid_attribute {}
