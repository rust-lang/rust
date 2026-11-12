//! Helper module for exporting the `view_types`/`view_projection` macros.

/// Creates a view type.
///
/// ```
/// #![feature(view_types)]
//
/// use std::view::view_type;
///
/// struct Foo {
///     bar: usize,
///     baz: u32,
/// }
///
/// type FooBar = view_type!(Foo.{ bar });
/// ```
#[allow_internal_unstable(internal_syntax)]
#[unstable(feature = "view_types", issue = "155938")]
#[diagnostic::opaque]
pub macro view_type($($arg:tt)*) {
    k#view_type($($arg)*)
}

/// Creates an instance of a view type.
///
/// ```
/// #![feature(view_types)]
///
/// use std::view::{view_type, view_projection};
//
/// struct Foo {
///     bar: usize,
///     baz: u32,
/// }
///
/// let foo = Foo { bar: 42, baz: 101 };
/// let foo_: &view_type!(Foo.{ bar }) = &view_projection!(foo.{ bar });
/// ```
#[allow_internal_unstable(internal_syntax)]
#[unstable(feature = "view_types", issue = "155938")]
#[diagnostic::opaque]
pub macro view_projection($($arg:tt)*) {
    k#view_projection($($arg)*)
}
