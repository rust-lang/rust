/// Creates a [`Vec`] containing the arguments.
///
/// `vec!` allows `Vec`s to be defined with the same syntax as array expressions.
/// There are two forms of this macro:
///
/// - Create a [`Vec`] containing a given list of elements:
///
/// ```
/// let v = vec![1, 2, 3];
/// assert_eq!(v[0], 1);
/// assert_eq!(v[1], 2);
/// assert_eq!(v[2], 3);
/// ```
///
/// - Create a [`Vec`] from a given element and size:
///
/// ```
/// let v = vec![1; 3];
/// assert_eq!(v, [1, 1, 1]);
/// ```
///
/// Note that unlike array expressions this syntax supports all elements
/// which implement [`Clone`] and the number of elements doesn't have to be
/// a constant.
///
/// This will use `clone` to duplicate an expression, so one should be careful
/// using this with types having a nonstandard `Clone` implementation. For
/// example, `vec![Rc::new(1); 5]` will create a vector of five references
/// to the same boxed integer value, not five references pointing to independently
/// boxed integers.
///
/// Also, note that `vec![expr; 0]` is allowed, and produces an empty vector.
/// This will still evaluate `expr`, however, and immediately drop the resulting value, so
/// be mindful of side effects.
///
/// [`Vec`]: crate::vec::Vec
#[cfg(not(no_global_oom_handling))]
#[macro_export]
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_diagnostic_item = "vec_macro"]
#[allow_internal_unstable(rustc_attrs, liballoc_internals)]
macro_rules! vec {
    () => (
        $crate::vec::Vec::new()
    );
    ($elem:expr; $n:expr) => (
        $crate::vec::from_elem($elem, $n)
    );
    ($($x:expr),+ $(,)?) => (
        <[_]>::into_vec(
            // Using the intrinsic produces a dramatic improvement in stack usage for
            // unoptimized programs using this code path to construct large Vecs.
            $crate::boxed::box_new([$($x),+])
        )
    );
}

/// Creates a `String` using interpolation of runtime expressions.
///
/// The first argument `format!` receives is a format string. This must be a string
/// literal. The power of the formatting string is in the `{}`s contained.
/// Additional parameters passed to `format!` replace the `{}`s within the
/// formatting string in the order given unless named or positional parameters
/// are used.
///
/// See [the formatting syntax documentation in `std::fmt`](../std/fmt/index.html)
/// for details.
///
/// A common use for `format!` is concatenation and interpolation of strings.
/// The same convention is used with [`print!`] and [`write!`] macros,
/// depending on the intended destination of the string; all these macros internally use [`format_args!`].
///
/// To convert a single value to a string, use the [`to_string`] method. This
/// will use the [`Display`] formatting trait.
///
/// To concatenate literals into a `&'static str`, use the [`concat!`] macro.
///
/// [`print!`]: ../std/macro.print.html
/// [`write!`]: core::write
/// [`format_args!`]: core::format_args
/// [`to_string`]: crate::string::ToString
/// [`Display`]: core::fmt::Display
/// [`concat!`]: core::concat
///
/// # Panics
///
/// `format!` panics if a formatting trait implementation returns an error.
/// This indicates an incorrect implementation
/// since `fmt::Write for String` never returns an error itself.
///
/// # Examples
///
/// ```
/// # #![allow(unused_must_use)]
/// format!("test");                             // => "test"
/// format!("hello {}", "world!");               // => "hello world!"
/// format!("x = {}, y = {val}", 10, val = 30);  // => "x = 10, y = 30"
/// let (x, y) = (1, 2);
/// format!("{x} + {y} = 3");                    // => "1 + 2 = 3"
/// ```
#[macro_export]
#[stable(feature = "rust1", since = "1.0.0")]
#[allow_internal_unstable(hint_must_use, liballoc_internals)]
#[rustc_diagnostic_item = "format_macro"]
macro_rules! format {
    ($($arg:tt)*) => {
        $crate::__export::must_use({
            let res = $crate::fmt::format($crate::__export::format_args!($($arg)*));
            res
        })
    }
}
