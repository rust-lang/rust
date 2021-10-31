//! The `Default` trait for types which may have meaningful default values.

#![stable(feature = "rust1", since = "1.0.0")]

/// A trait for giving a type a useful default value.
///
/// Sometimes, you want to fall back to some kind of default value, and
/// don't particularly care what it is. This comes up often with `struct`s
/// that define a set of options:
///
/// ```
/// # #[allow(dead_code)]
/// struct SomeOptions {
///     foo: i32,
///     bar: f32,
/// }
/// ```
///
/// How can we define some default values? You can use `Default`:
///
/// ```
/// # #[allow(dead_code)]
/// #[derive(Default)]
/// struct SomeOptions {
///     foo: i32,
///     bar: f32,
/// }
///
/// fn main() {
///     let options: SomeOptions = Default::default();
/// }
/// ```
///
/// Now, you get all of the default values. Rust implements `Default` for various primitives types.
///
/// If you want to override a particular option, but still retain the other defaults:
///
/// ```
/// # #[allow(dead_code)]
/// # #[derive(Default)]
/// # struct SomeOptions {
/// #     foo: i32,
/// #     bar: f32,
/// # }
/// fn main() {
///     let options = SomeOptions { foo: 42, ..Default::default() };
/// }
/// ```
///
/// ## Derivable
///
/// This trait can be used with `#[derive]` if all of the type's fields implement
/// `Default`. When `derive`d, it will use the default value for each field's type.
///
/// ## How can I implement `Default`?
///
/// Provide an implementation for the `default()` method that returns the value of
/// your type that should be the default:
///
/// ```
/// # #![allow(dead_code)]
/// enum Kind {
///     A,
///     B,
///     C,
/// }
///
/// impl Default for Kind {
///     fn default() -> Self { Kind::A }
/// }
/// ```
///
/// # Examples
///
/// ```
/// # #[allow(dead_code)]
/// #[derive(Default)]
/// struct SomeOptions {
///     foo: i32,
///     bar: f32,
/// }
/// ```
#[cfg_attr(not(test), rustc_diagnostic_item = "Default")]
#[stable(feature = "rust1", since = "1.0.0")]
pub trait Default: Sized {
    /// Returns the "default value" for a type.
    ///
    /// Default values are often some kind of initial value, identity value, or anything else that
    /// may make sense as a default.
    ///
    /// # Examples
    ///
    /// Using built-in default values:
    ///
    /// ```
    /// let i: i8 = Default::default();
    /// let (x, y): (Option<String>, f64) = Default::default();
    /// let (a, b, (c, d)): (i32, u32, (bool, bool)) = Default::default();
    /// ```
    ///
    /// Making your own:
    ///
    /// ```
    /// # #[allow(dead_code)]
    /// enum Kind {
    ///     A,
    ///     B,
    ///     C,
    /// }
    ///
    /// impl Default for Kind {
    ///     fn default() -> Self { Kind::A }
    /// }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn default() -> Self;
}

/// Return the default value of a type according to the `Default` trait.
///
/// The type to return is inferred from context; this is equivalent to
/// `Default::default()` but shorter to type.
///
/// For example:
/// ```
/// #![feature(default_free_fn)]
///
/// use std::default::default;
///
/// #[derive(Default)]
/// struct AppConfig {
///     foo: FooConfig,
///     bar: BarConfig,
/// }
///
/// #[derive(Default)]
/// struct FooConfig {
///     foo: i32,
/// }
///
/// #[derive(Default)]
/// struct BarConfig {
///     bar: f32,
///     baz: u8,
/// }
///
/// fn main() {
///     let options = AppConfig {
///         foo: default(),
///         bar: BarConfig {
///             bar: 10.1,
///             ..default()
///         },
///     };
/// }
/// ```
#[unstable(feature = "default_free_fn", issue = "73014")]
#[must_use]
#[inline]
pub fn default<T: Default>() -> T {
    Default::default()
}

/// Derive macro generating an impl of the trait `Default`.
#[rustc_builtin_macro(Default, attributes(default))]
#[stable(feature = "builtin_macro_prelude", since = "1.38.0")]
#[allow_internal_unstable(core_intrinsics)]
pub macro Default($item:item) {
    /* compiler built-in */
}

macro_rules! default_impl {
    ($t:ty, $v:expr, $doc:tt) => {
        #[stable(feature = "rust1", since = "1.0.0")]
        #[rustc_const_unstable(feature = "const_default_impls", issue = "87864")]
        impl const Default for $t {
            #[inline]
            #[doc = $doc]
            fn default() -> $t {
                $v
            }
        }
    };
}

default_impl! { (), (), "Returns the default value of `()`" }
default_impl! { bool, false, "Returns the default value of `false`" }
default_impl! { char, '\x00', "Returns the default value of `\\x00`" }

default_impl! { usize, 0, "Returns the default value of `0`" }
default_impl! { u8, 0, "Returns the default value of `0`" }
default_impl! { u16, 0, "Returns the default value of `0`" }
default_impl! { u32, 0, "Returns the default value of `0`" }
default_impl! { u64, 0, "Returns the default value of `0`" }
default_impl! { u128, 0, "Returns the default value of `0`" }

default_impl! { isize, 0, "Returns the default value of `0`" }
default_impl! { i8, 0, "Returns the default value of `0`" }
default_impl! { i16, 0, "Returns the default value of `0`" }
default_impl! { i32, 0, "Returns the default value of `0`" }
default_impl! { i64, 0, "Returns the default value of `0`" }
default_impl! { i128, 0, "Returns the default value of `0`" }

default_impl! { f32, 0.0f32, "Returns the default value of `0.0`" }
default_impl! { f64, 0.0f64, "Returns the default value of `0.0`" }
