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
/// In the following example, the returned [`Result`] is the only sign that writing the message
/// might have failed:
///
/// ```
/// # #![allow(unused_must_use)]
/// fn write_message() -> std::io::Result<()> {
///     // Write the message...
///     Ok(())
/// }
///
/// write_message();
/// ```
///
/// Ignoring that [`Result`] triggers this warning:
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
/// ```
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
/// ```
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
/// ```
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
/// [`unused_must_use`]: ../rustc/lints/listing/warn-by-default.html#unused-must-use
/// [the `must_use` attribute]: ../reference/attributes/diagnostics.html#the-must_use-attribute
mod must_use_attribute {}

#[doc(attribute = "allow")]
//
/// The `allow` attribute suppresses lint diagnostics that would otherwise produce
/// warnings or errors. It can be used on any lint or lint group (except those
/// set to [`forbid`]).
///
/// ```
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
/// ```
/// #[allow(unused_variables, unused_mut)]
/// fn main() {
///     let mut x: u32 = 42;
/// }
/// ```
///
/// This is mostly used to prevent lint warnings or errors while still under development.
///
/// It cannot override a lint that has been set to [`forbid`].
///
/// It's also important to consider that overusing `allow` could make code harder to maintain
/// and possibly hide issues. To mitigate this issue, using the `expect` attribute is preferred.
///
/// `allow` can be overridden by [`warn`], [`deny`], and [`forbid`].
///
/// The lint checks supported by rustc can be found via `rustc -W help`,
/// along with their default settings and are documented in [the `rustc` book].
///
/// [the `rustc` book]: ../rustc/lints/listing/index.html
///
/// For more information, see the Reference on [the `allow` attribute].
///
/// [the `allow` attribute]: ../reference/attributes/diagnostics.html#lint-check-attributes
/// [`forbid`]: ./attribute.forbid.html
/// [`warn`]: ./attribute.warn.html
/// [`deny`]: ./attribute.deny.html
mod allow_attribute {}

#[doc(attribute = "cfg")]
//
/// Used for conditional compilation.
///
/// The `cfg` attribute allows compiling an item under specific conditions, otherwise it
/// will be ignored.
///
/// ```
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
/// Conditions can also be combined with `all(...)`, `any(...)`, and `not(...)`:
///
/// * `all`: True if all given predicates are true.
/// * `any`: True if at least one of the given predicates is true.
/// * `not`: True if the predicate is false and false if the predicate is true.
///
/// ```
/// #[cfg(all(unix, target_pointer_width = "64"))]
/// fn unix_64bit() {
///     // ...
/// }
/// ```
///
/// If you want to use this mechanism in an [`if`] condition in your code, you
/// can use the [`cfg!`] macro. To conditionally apply an attribute,
/// see [`cfg_attr`].
///
/// For more information, see the Reference on [the `cfg` attribute].
///
/// [`cfg_attr`]: ../reference/conditional-compilation.html#the-cfg_attr-attribute
/// [the `cfg` attribute]: ../reference/conditional-compilation.html#the-cfg-attribute
/// [`if`]: ./keyword.if.html
mod cfg_attribute {}

#[doc(attribute = "deny")]
//
/// Emits an error, preventing the compilation from finishing, when a lint check has failed.
/// This is useful for enforcing rules or preventing certain patterns:
///
/// ```compile_fail
/// #[deny(unused)]
/// fn foo() {
///     let x = 42; // Emits an error because x is unused.
/// }
/// ```
///
/// `deny` can be overridden by [`allow`], [`warn`], and [`forbid`]:
///
/// ```
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
/// ```compile_fail
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
/// [`forbid`]: ./attribute.forbid.html
/// [`allow`]: ./attribute.allow.html
/// [`warn`]: ./attribute.warn.html
/// [`deny`]: ./attribute.deny.html
mod deny_attribute {}

#[doc(attribute = "forbid")]
//
/// Emits an error, preventing the compilation from finishing, when a lint check has failed.
///
/// A lint set to `forbid` cannot be overridden by [`allow`] or [`warn`].
/// Attempting either will result in a compilation error. Writing `#[deny(...)]` on the same lint inside a
/// `forbid` scope is permitted, but has no effect; the lint remains at the `forbid` level.
///
/// This is useful for enforcing strict policies that should not be relaxed
/// anywhere in the codebase. Example:
///
/// ```
/// #![forbid(unsafe_code)]
///
/// // This would cause a compilation error if uncommented:
/// // #[allow(unsafe_code)] // error: cannot override `forbid`
/// ```
///
/// Multiple lints can be set to `forbid` at once:
///
/// ```
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
/// [`allow`]: ./attribute.allow.html
/// [`warn`]: ./attribute.warn.html
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
/// ```
/// #[deprecated(since = "1.0.0", note = "Use bar instead")]
/// struct Foo;
/// struct Bar;
/// ```
///
/// The `deprecated` attribute helps developers transition away from old code by providing warnings
/// when deprecated items are used. Note that during `Cargo` builds, warnings on dependencies get
/// silenced by default, so you may not see a deprecation warning unless you build that dependency
/// directly.
///
/// For more information, see the Reference on [the `deprecated` attribute].
///
/// [the `deprecated` attribute]: ../reference/attributes/diagnostics.html#the-deprecated-attribute
mod deprecated_attribute {}

#[doc(attribute = "warn")]
//
/// Emits a warning during compilation when a lint check failed.
///
/// Unlike [`deny`] or [`forbid`], `warn` does not produce a hard error: the compilation
/// continues, but the compiler emits a warning message. `warn` can be overridden by [`allow`],
/// [`deny`], and [`forbid`].
///
/// Example:
///
/// ```compile_fail
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
/// mainly useful for lints that are normally [`allow`] by default.
///
/// Multiple lints can be set to `warn` at once:
///
/// ```compile_fail
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
/// [`allow`]: ./attribute.allow.html
/// [`deny`]: ./attribute.deny.html
/// [`forbid`]: ./attribute.forbid.html
mod warn_attribute {}

#[doc(attribute = "no_std")]
//
/// Prevents automatically linking the standard library.
///
/// Written as an inner attribute at the top of the crate root, the `no_std` attribute stops
/// the compiler from linking [`std`] into the crate, and only links [`core`].
///
/// The attribute also swaps which prelude gets inserted, with the [`core`] prelude replacing
/// the [`std`] prelude. Items like [`Option`], [`Result`], and the primitive types remain
/// available from [`core`] without any imports needed:
///
/// ```rust,ignore (no_std)
/// #![no_std]
///
/// fn halve(x: u32) -> Option<u32> {
///     if x % 2 == 0 { Some(x / 2) } else { None }
/// }
/// ```
///
/// Anything that requires heap memory allocations is not part of [`core`]. Linking the [`alloc`]
/// crate explicitly can be used to include these types:
///
/// ```rust,ignore (no_std + needs a global allocator to link)
/// #![no_std]
///
/// extern crate alloc;
///
/// use alloc::vec::Vec;
/// ```
///
/// A `no_std` binary also removes the startup routine and default panic handler `std`
/// normally provides. It must define its own `#[panic_handler]`, and typically its own
/// entry point by additionally using `#![no_main]`:
///
/// ```rust,ignore (no_std + needs more than no_main alone to link)
/// #![no_std]
/// #![no_main]
///
/// use core::panic::PanicInfo;
///
/// #[panic_handler]
/// fn on_panic(_info: &PanicInfo) -> ! {
///     loop {}
/// }
/// ```
///
/// For more information, see the Reference on [the `no_std` attribute].
///
/// [`std`]: ../std/index.html
/// [`core`]: ../core/index.html
/// [`alloc`]: ../alloc/index.html
/// [`Option`]: option::Option
/// [`Result`]: result::Result
/// [the `no_std` attribute]: ../reference/names/preludes.html#the-no_std-attribute
mod no_std_attribute {}

#[doc(attribute = "inline")]
//
/// Suggest that the compiler inline a function at its call sites.
///
/// Inlining replaces a call with a copy of the called function's body, which can remove the
/// overhead of the call. The `inline` attribute is only a hint: the compiler may ignore it, and
/// it already inlines functions on its own when that looks worthwhile. Poor choices about what to
/// inline can make a program larger or slower.
///
/// Where it does matter is inlining across crate boundaries. A non-generic function is not
/// normally inlined into another crate, since the calling crate compiles against only its
/// signature. Marking it `#[inline]` makes the body available to other crates so they can inline
/// it too:
///
/// ```rust
/// # #![allow(dead_code)]
/// #[inline]
/// pub fn square(x: i32) -> i32 {
///     x * x
/// }
/// ```
///
/// Generic functions do not need this. They are instantiated in each crate that uses them, so
/// their bodies are already available to inline.
///
/// The attribute applies to functions and has three forms:
///
/// - `#[inline]` suggests inlining the function.
/// - `#[inline(always)]` suggests inlining it at every call site.
/// - `#[inline(never)]` suggests never inlining it.
///
/// You should almost never need `#[inline(always)]`: prefer to let the compiler decide unless
/// profiling shows a small, hot function that benefits from it. `#[inline(never)]` is useful to
/// keep a rarely used path, such as a function that only reports an error, out of its caller.
///
/// For more information, see the Reference on [the `inline` attribute].
///
/// [the `inline` attribute]: ../reference/attributes/codegen.html#the-inline-attribute
mod inline_attribute {}

#[doc(attribute = "cold")]
//
/// Hint to the compiler that a function is unlikely to be called.
///
/// Marking a function `#[cold]` tells the compiler that calls to it are rare, so it can
/// optimize for the common case where the function is not called. It is only a hint: the
/// compiler may ignore it, and it does not change the function's behavior.
///
/// It is typically used on functions that handle uncommon cases, such as error or panic paths:
///
/// ```rust
/// # #![allow(dead_code)]
/// fn check(value: i32) {
///     if value < 0 {
///         report_error("value must be non-negative");
///     }
///     // ... the common case continues here ...
/// }
///
/// #[cold]
/// fn report_error(message: &str) {
///     eprintln!("error: {message}");
/// }
/// ```
///
/// For more information, see the Reference on [the `cold` attribute].
///
/// [the `cold` attribute]: ../reference/attributes/codegen.html#the-cold-attribute
mod cold_attribute {}

#[doc(attribute = "track_caller")]
//
/// Make a function report the location of its caller instead of its own.
///
/// When a function panics, the panic message normally points at the line inside that function
/// where the panic happened. `#[track_caller]` changes that: it lets the function see the
/// [`Location`] it was called from, so the panic (and any direct use of [`Location::caller`])
/// points at the call site instead. The standard library uses this on methods like
/// [`Option::unwrap`], so a failed `unwrap` blames the line that called it rather than a line
/// inside the standard library.
///
/// ```rust,should_panic
/// #[track_caller]
/// fn assert_even(n: i32) {
///     assert!(n % 2 == 0, "{n} is not even");
/// }
///
/// // The panic blames this line, not the `assert!` inside `assert_even`.
/// assert_even(3);
/// ```
///
/// The attribute applies to functions with the default `"Rust"` ABI, other than `fn main`.
///
/// For more information, see the Reference on [the `track_caller` attribute].
///
/// [`Location`]: panic::Location
/// [`Location::caller`]: panic::Location::caller
/// [`Option::unwrap`]: Option::unwrap
/// [the `track_caller` attribute]: ../reference/attributes/codegen.html#the-track_caller-attribute
mod track_caller_attribute {}

#[doc(attribute = "proc_macro")]
//
/// Defines a function-like procedural macro.
///
/// Applied to a `pub` function at the root of a proc-macro crate, `proc_macro` makes that function usable as a macro invoked as
/// `foo!(...)` in other crates. The function receives the tokens written inside the invocation as a [`TokenStream`] and returns
/// the [`TokenStream`] that replaces the invocation:
///
/// ```rust, ignore (requires depending on the proc-macro crate)
/// # extern crate proc_macro;
/// use proc_macro::TokenStream;
///
/// #[proc_macro]
/// pub fn foo(input: TokenStream) -> TokenStream {
///    "fn answer() -> u32 { 67 }".parse().unwrap()
/// }
/// ```
///
/// The macro can only be invoked from other crates, not from the crate where it is defined:
///
/// ```rust,ignore (requires depending on the proc-macro crate)
/// use my_macro_crate::foo;
///
/// // Expands to `fn answer() -> u32 { 67 }`.
/// foo!();
///
/// fn main() {
///    println!("{}", answer()); // Prints 67
/// }
/// ```
///
/// The attribute is only usable with crates of the `proc-macro` crate type, which is set in the crate's `Cargo.toml`
/// with `proc-macro = true` in the `[lib]` section. Using it anywhere else is a compilation error:
///
/// ```text
///error: the `#[proc_macro]` attribute is only usable with crates of the `proc-macro` crate type
/// --> src/lib.rs:4:1
///  |
/// 4| #[proc_macro]
///  | ^^^^^^^^^^^^
/// ```
///
/// For more information, see the Reference on [function-like procedural macros] and the [`proc_macro`] crate documentation.
///
/// [`TokenStream`]: ../proc_macro/struct.TokenStream.html
/// [function-like procedural macros]: ../reference/procedural-macros.html#the-proc_macro-attribute
/// [`proc_macro`]: ../proc_macro/index.html
mod proc_macro_attribute {}

#[doc(attribute = "link_section")]
//
/// Places a function or static in a specific object-file section.
///
/// The `link_section` attribute specifies the section of the generated object file where a
/// function or static is placed. Section names and their meaning are target-specific.
///
/// ```rust,no_run
/// # #[cfg(target_os = "linux")] {
/// #[unsafe(link_section = ".example_section")]
/// pub static VALUE: u32 = 42;
/// # }
/// ```
///
/// Incorrectly placing code or data in a section can violate requirements imposed by the target,
/// linker, or runtime. For example, placing mutable data in a read-only section may result in
/// undefined behavior. For this reason, `link_section` is an unsafe attribute.
///
/// Starting with the 2024 edition, the attribute must be written using the `unsafe(...)` syntax.
/// Earlier editions also permit `#[link_section = "..."]`.
///
/// For more information, see the Reference on [the `link_section` attribute].
///
/// [the `link_section` attribute]: ../reference/abi.html#the-link_section-attribute
mod link_section_attribute {}
