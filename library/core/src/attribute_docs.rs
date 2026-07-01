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
/// The `allow` attribute suppresses lint diagnostics that would otherwise produce
/// warnings or errors. It can be used on any lint or lint group (except those
/// set to `forbid`).
///
/// ```rust
/// #[allow(dead_code)]
/// fn unused_function() {
///     // ...
/// }
///
/// fn main() {
///   // `unused_function` does not generate a compiler warning.
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
/// ```
///
/// Multiple lints can be set to `allow` at once with commas:
///
/// ```rust
/// #[allow(unused_variables, unused_mut)]
/// fn main() {
///     let mut x: u32 = 42;
/// }
/// ```
///
/// This is mostly used to prevent lint warnings or errors while still under development.
///
/// It cannot override a lint that has been set to `forbid`.
///
/// It's also important to consider that overusing `allow` could make code harder to maintain
/// and possibly hide issues. To mitigate this issue, using the `expect` attribute is preferred.
///
/// `allow` can be overridden by `warn`, `deny`, and `forbid`.
///
/// The lint checks supported by rustc can be found via `rustc -W help`,
/// along with their default settings and are documented in [the `rustc` book].
///
/// [the `rustc` book]: ../rustc/lints/listing/index.html
///
/// For more information, see the Reference on [the `allow` attribute].
///
/// [the `allow` attribute]: ../reference/attributes/diagnostics.html#lint-check-attributes
mod allow_attribute {}

#[doc(attribute = "cfg")]
//
/// Used for conditional compilation.
///
/// The `cfg` attribute allows compiling an item under specific conditions, otherwise it
/// will be ignored.
///
/// ```rust
/// // Only compiles this function for Linux.
/// #[cfg(target_os = "linux")]
/// fn platform_specific() {
///     println!("Running on Linux");
/// }
///
/// // Only compiles this function if not for Linux.
/// #[cfg(not(target_os = "linux"))]
/// fn platform_specific() {
///     println!("Running on something else");
/// }
/// ```
///
/// Depending on the platform you're targeting, only one of these two functions will be considered
/// during the compilation.
///
/// Conditions can also be combined with `all(...)`, `any(...)`, and `not(...)`.
///
/// * `all`: True if all given predicates are true.
/// * `any`: True if at least one of the given predicates is true.
/// * `not`: True if the predicate is false and false if the predicate is true.
///
/// ```rust
/// #[cfg(all(unix, target_pointer_width = "64"))]
/// fn unix_64bit() {
/// }
/// ```
///
/// If you want to use this mechanism in an `if` condition in your code, you
/// can use the [`cfg!`] macro. To conditionally apply an attribute,
/// see [`cfg_attr`].
///
/// For more information, see the Reference on [the `cfg` attribute].
///
/// [`cfg_attr`]: ../reference/conditional-compilation.html#the-cfg_attr-attribute
/// [the `cfg` attribute]: ../reference/conditional-compilation.html#the-cfg-attribute
mod cfg_attribute {}

#[doc(attribute = "cfg_attr")]
//
/// Used for conditional application of attributes.
///
/// The `cfg_attr` attribute applies one or more attributes to an item only if a given
/// configuration predicate is true. If the condition is false, the attributes are ignored.
///
/// Example:
///
/// ```rust
/// // This function is annotated with `#[test]` only when testing is active.
/// #[cfg_attr(test, test)]
/// fn my_function() {
///     // ...
/// }
/// ```
///
/// You can apply multiple attributes by grouping them in parentheses:
///
/// ```rust
/// #[cfg_attr(feature = "nightly", allow(dead_code))]
/// // This function only gets the `allow(dead_code)` attribute when `feature = "nightly"` is
/// // active.
/// fn nightly_only_function() {
///     // ...
/// }
/// ```
///
/// The predicate uses the same syntax as [`cfg`]. For complex conditions, you can combine
/// `all(...)`, `any(...)`, and `not(...)` just like with `cfg`.
///
/// For more information, see the Reference on [the `cfg_attr` attribute].
///
/// [the `cfg_attr` attribute]: ../reference/conditional-compilation.html#the-cfg_attr-attribute
mod cfg_attr_attribute {}

#[doc(attribute = "deny")]
//
/// Emits an error, preventing the compilation from finishing, when a lint check has failed.
/// This is useful for enforcing rules or preventing certain patterns:
///
/// ```rust,compile_fail
/// #[deny(unused)]
/// fn foo() {
///     let x = 42; // Emits an error because x is unused.
/// }
/// ```
///
/// `deny` can be overridden by `allow`, `warn`, and `forbid`:
///
/// ```rust
/// #![deny(unused)]
///
/// #[allow(unused)] // We override the `deny` for this function.
/// fn foo() {
///     let x = 42; // No lint emitted even though `x` is unused.
/// }
/// ```
///
/// Multiple lints can also be set to `deny` at once:
///
/// ```rust,compile_fail
/// #![deny(unused_imports, unused_variables)]
/// use std::collections::*;
///
/// fn main() {
///     let mut x = 10;
/// }
/// ```
///
/// The lint checks supported by rustc can be found via `rustc -W help`,
/// along with their default settings and are documented in [the `rustc` book].
///
/// [the `rustc` book]: ../rustc/lints/listing/index.html
///
/// For more information, see the Reference on [the `deny` attribute].
///
/// [the `deny` attribute]: ../reference/attributes/diagnostics.html#lint-check-attributes
mod deny_attribute {}

#[doc(attribute = "forbid")]
//
/// Emits an error, preventing the compilation from finishing, when a lint check has failed.
///
/// A lint set to `forbid` cannot be overridden by `allow` or `warn`.
/// Attempting either will result in a compilation error. Writing `#[deny(...)]` on the same lint inside a
/// `forbid` scope is permitted, but has no effect; the lint remains at the `forbid` level.
///
/// This is useful for enforcing strict policies that should not be relaxed
/// anywhere in the codebase. Example:
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
/// The lint checks supported by rustc can be found via `rustc -W help`,
/// along with their default settings and are documented in [the `rustc` book].
///
/// [the `rustc` book]: ../rustc/lints/listing/index.html
///
/// For more information, see the Reference on [the `forbid` attribute].
///
/// [the `forbid` attribute]: ../reference/attributes/diagnostics.html#lint-check-attributes
mod forbid_attribute {}

#[doc(attribute = "deprecated")]
//
/// Emits a warning during compilation when an item with this attribute is used.
/// `since` and `note` are optional fields giving more detail about why the item is deprecated.
///
/// * `since`: the version since when the item is deprecated.
/// * `note`: the reason why an item is deprecated.
///
/// Example:
///
/// ```rust
/// #[deprecated(since = "1.0.0", note = "Use bar instead")]
/// struct Foo;
/// struct Bar;
/// ```
///
/// `deprecated` attribute helps developers transition away from old code by providing warnings when
/// deprecated items are used. Note that during `Cargo` builds, warnings on dependencies get silenced
/// by default, so you may not see a deprecation warning unless you build that dependency directly.
///
/// For more information, see the Reference on [the `deprecated` attribute].
///
/// [the `deprecated` attribute]: ../reference/attributes/diagnostics.html#the-deprecated-attribute
mod deprecated_attribute {}

#[doc(attribute = "warn")]
//
/// Emits a warning during compilation when a lint check failed.
///
/// Unlike `deny` or `forbid`, `warn` does not produce a hard error: the compilation continues, but
/// the compiler emits a warning message. `warn` can be overridden by `allow`, `deny`, and `forbid`.
///
/// Example:
///
/// ```rust,compile_fail
/// #![allow(unused)]
///
/// #[warn(unused)] // We override the allowed `unused` lint.
/// fn foo() {
///     // This lint warns by default even without #[warn(unused)] being explicitly set
///     let x = 42; // warning: unused variable `x`
/// }
/// ```
///
///
/// Many lints, including `unused`, are already set to `warn` by default so this attribute is
/// mainly useful for lints that are normally `allow` by default.
///
/// Multiple lints can be set to `warn` at once:
///
/// ```rust,compile_fail
/// #[warn(unused_mut, unused_variables)]
/// fn main() {
///     let mut x = 42;
/// }
/// ```
///
/// The lint checks supported by rustc can be found via `rustc -W help`,
/// along with their default settings and are documented in [the `rustc` book].
///
/// [the `rustc` book]: ../rustc/lints/listing/index.html
///
/// For more information, see the Reference on [the `warn` attribute].
///
/// [the `warn` attribute]: ../reference/attributes/diagnostics.html#lint-check-attributes
mod warn_attribute {}

#[doc(attribute = "used")]
//
/// The `#[used]` attribute can be applied to static variables to prevent the Rust compiler
/// from optimizing them away even if they are not referenced anywhere else in the code.
///
/// It tells the compiler that an item is still in use or needed elsewhere and, because of this,
/// it is kept in the output object.
///
/// Example:
///
/// ```rust
/// #[used]
/// static FOO: u16 = 42; // FOO is not optimized away as dead code.
///
/// static BAR: u16 = 12; // BAR may be optimized away.
/// fn main() {
/// }
/// ```
/// To confirm, we compile this program into an object file, you'll see that `FOO` makes it to the
/// object file but `BAR` doesn't. Neither static variable is used by the program.
///
/// ```text
/// rustc -C opt-level=3 --emit=obj used.rs
/// nm -C used.o
/// 0000000000000000 T main
///                  U std::rt::lang_start_internal
/// 0000000000000000 T std::rt::lang_start
/// 0000000000000000 t std::rt::lang_start::{{closure}}
/// 0000000000000000 t std::sys::backtrace::__rust_begin_short_backtrace
/// 0000000000000000 t core::ops::function::FnOnce::call_once{{vtable.shim}}
/// 0000000000000000 r used::FOO
/// 0000000000000000 T used::main
/// ```
///
/// For more information, see the Reference on [the `used` attribute].
///
/// [the `used` attribute]: ../reference/abi.html#the-used-attribute
mod used_attribute {}

#[doc(attribute = "ignore")]
//
/// The `ignore` attribute is used with the `test` attribute to stop the test harness from
/// executing a function as a test. The `ignore` attribute may only be applied to functions
/// annotated with the test attribute.
///
/// Example:
///
/// ```rust
/// #[test]
/// #[ignore]
/// fn test() {
///   // ... This test is ignored by the test harness (the compiler still compiles it).
/// }
/// ```
///
/// The `ignore` attribute can be very useful when a test is incomplete and we need to compile the
/// code.
///
/// To run ignored tests, use `cargo test -- --ignored`.
///
/// For more information, see the Reference on [the `ignore` attribute].
///
/// [the `ignore` attribute]: ../reference/attributes/testing.html#the-ignore-attribute
mod ignore_attribute {}

#[doc(attribute = "expect")]
//
/// The `#[expect]` attribute declares that a particular lint is expected to be emitted.
/// If the lint would be emitted, it is suppressed and the expectation is fulfilled.
/// If the lint is not emitted at all, the expectation is unfulfilled and a warning is generated.
///
/// `expect` can be overridden by `warn` or `allow` in an inner scope leaving the outer `expect` attribute
/// unfulfilled.
///
/// Example:
///
/// ```rust
/// fn main() {
///     #[expect(unused_variables)]
///     let name = "rust-lang"; // The `unused_variables` warning is suppressed.
/// }
/// ```
///
/// Multiple lints can be set to `expect` at once, each one is expected separately. For a lint group, it’s enough
/// if one lint inside the group has been emitted:
///
/// ```rust,compile_fail
/// #[expect(unused_mut, unused_variables)]
/// fn main() {
///     // This attribute creates two lint expectations. The `unused_mut` lint will be
///     // suppressed and with that fulfill the first expectation. The `unused_variables`
///     // wouldn't be emitted, since the variable is used. That expectation will therefore
///     // be unsatisfied, and a warning will be emitted.
///     let mut x = 42;
///     println!("x: {x}");
/// }
/// ```
///
/// For more information, see the Reference on [the `expect` attribute].
///
/// [the `expect` attribute]: ../reference/attributes/diagnostics.html#lint-check-attributes
mod expect_attribute {}

#[doc(attribute = "should_panic")]
//
/// The `should_panic` attribute is used with the `test` attribute to pass a test that is expected
/// to panic.
///
/// The `should_panic` attribute may only be applied to functions annotated with the test attribute.
///
/// ```rust
/// fn add(a: i32, b: i32) -> i32 {
///     a + b
/// }
///
/// fn main() {
///   // ...
/// }
///
/// #[cfg(test)]
/// mod tests {
///     use super::*;
///     #[test]
///     fn test_add() {
///         assert_eq!(add(2,3), 5); // This test passes.
///     }
///
///     #[test]
///     #[should_panic]
///     fn test_add_neg() {
///         assert_eq!(add(-1, 1), 10); // Panics (0 != 10), so the test passes.
///     }
///
///     #[test]
///     #[should_panic]
///     fn failing_test() {
///         panic!("This test is meant to panic"); // This test passes.
///     }
/// }
/// ```
/// For more information, see the Reference on [the `should_panic` attribute].
///
/// [the `should_panic` attribute]: ../reference/attributes/testing.html#the-should_panic-attribute
mod should_panic_attribute {}

#[doc(attribute = "path")]
//
/// The `#[path = "..."]` attribute tells the compiler where to find a module's source file, overriding the default
/// file lookup.
///
/// Example:
///
/// ```rust,compile_fail
/// mod foo; // By default `rustc` loads the source from `foo.rs`.
/// ```
///
/// `path` overrides this behavior:
///
/// ```rust,compile_fail
/// #[path = "bar.rs"]
/// mod foo; // Now bar.rs is loaded as foo.
/// ```
///
/// You can combine `#[path]` with `#[cfg]` to load platform-specific files:
///
/// ```rust,compile_fail
/// // On Linux, the platform/linux.rs source file is loaded.
/// #[cfg(target_os = "linux")]
/// #[path = "platform/linux.rs"]
/// mod platform;
///
/// // On Windows, the platform/windows.rs source file is loaded.
/// #[cfg(target_os = "windows")]
/// #[path = "platform/windows.rs"]
/// mod platform;
/// ```
///
/// For more information, see the Reference on [the `path` attribute].
///
/// [the `path` attribute]: ../reference/items/modules.html#the-path-attribute
mod path_attribute {}
