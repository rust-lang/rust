/// Creates a new closure that returns an iterator where each iteration steps the given
/// generator to the next `yield` statement.
///
/// Similar to [`iter::from_fn`], but allows arbitrary control flow.
///
/// [`iter::from_fn`]: crate::iter::from_fn
///
/// # Examples
///
/// ```
/// #![feature(iter_macro, coroutines)]
///
/// let it = std::iter::iter!{|| {
///     yield 1;
///     yield 2;
///     yield 3;
/// } }();
/// let v: Vec<_> = it.collect();
/// assert_eq!(v, [1, 2, 3]);
/// ```
#[unstable(feature = "iter_macro", issue = "none", reason = "generators are unstable")]
#[allow_internal_unstable(coroutines, iter_from_coroutine)]
#[rustc_builtin_macro]
pub macro iter($($t:tt)*) {
    /* compiler-builtin */
}
