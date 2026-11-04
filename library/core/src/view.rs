//! Helpers module for exporting the `view_types` macro.

/// Creates a view type.
/// ```
/// #![feature(view_types)]
//
/// struct Foo {
///     bar: usize,
///     baz: u32,
/// }
///
/// type FooBar = std::view::view_type!(Foo.{ bar });
/// ```
#[allow_internal_unstable(internal_syntax)]
#[unstable(feature = "view_types", issue = "155938")]
#[diagnostic::opaque]
pub macro view_type($($arg:tt)*) {
    k#view_type($($arg)*)
}
