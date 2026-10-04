//! Helpers module for exporting the `view_types` macro.

/// Creates a view type.
/// ```
/// #![feature(view_types, view_type_macro)]
//
/// struct Foo {
///     bar: usize,
///     baz: u32,
/// }
///
/// type FooBar = std::view::view_type!(Foo.{ bar });
/// ```
#[macro_export]
#[rustc_builtin_macro(view_type)]
#[unstable(feature = "view_type_macro", issue = "155938")]
macro_rules! view_type {
    ($($arg:tt)*) => {
        /* compiler built-in */
    };
}
