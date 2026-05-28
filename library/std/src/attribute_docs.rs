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
