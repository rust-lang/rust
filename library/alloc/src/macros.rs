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
/// [`Vec`]: crate::vec::Vec
#[cfg(not(test))]
#[macro_export]
#[stable(feature = "rust1", since = "1.0.0")]
#[allow_internal_unstable(box_syntax)]
macro_rules! vec {
    () => (
        $crate::vec::Vec::new()
    );
    ($elem:expr; $n:expr) => (
        $crate::vec::from_elem($elem, $n)
    );
    ($($x:expr),+ $(,)?) => (
        <[_]>::into_vec(box [$($x),+])
    );
}

// HACK(japaric): with cfg(test) the inherent `[T]::into_vec` method, which is
// required for this macro definition, is not available. Instead use the
// `slice::into_vec`  function which is only available with cfg(test)
// NB see the slice::hack module in slice.rs for more information
#[cfg(test)]
macro_rules! vec {
    () => (
        $crate::vec::Vec::new()
    );
    ($elem:expr; $n:expr) => (
        $crate::vec::from_elem($elem, $n)
    );
    ($($x:expr),*) => (
        $crate::slice::into_vec(box [$($x),*])
    );
    ($($x:expr,)*) => (vec![$($x),*])
}

/// Creates a `String` using interpolation of runtime expressions.
///
/// The first argument `format!` receives is a format string. This must be a string
/// literal. The power of the formatting string is in the `{}`s contained.
///
/// Additional parameters passed to `format!` replace the `{}`s within the
/// formatting string in the order given unless named or positional parameters
/// are used; see [`std::fmt`] for more information.
///
/// A common use for `format!` is concatenation and interpolation of strings.
/// The same convention is used with [`print!`] and [`write!`] macros,
/// depending on the intended destination of the string.
///
/// To convert a single value to a string, use the [`to_string`] method. This
/// will use the [`Display`] formatting trait.
///
/// [`std::fmt`]: ../std/fmt/index.html
/// [`print!`]: ../std/macro.print.html
/// [`write!`]: core::write
/// [`to_string`]: crate::string::ToString
/// [`Display`]: core::fmt::Display
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
/// format!("test");
/// format!("hello {}", "world!");
/// format!("x = {}, y = {y}", 10, y = 30);
/// ```
#[macro_export]
#[stable(feature = "rust1", since = "1.0.0")]
macro_rules! format {
    ($($arg:tt)*) => {{
        let res = $crate::fmt::format($crate::__export::format_args!($($arg)*));
        res
    }}
}
