//! impl bool {}

use crate::marker::Destruct;

impl bool {
    /// Returns `Some(t)` if the `bool` is [`true`](../std/keyword.true.html),
    /// or `None` otherwise.
    ///
    /// Arguments passed to `then_some` are eagerly evaluated; if you are
    /// passing the result of a function call, it is recommended to use
    /// [`then`], which is lazily evaluated.
    ///
    /// [`then`]: bool::then
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!(false.then_some(0), None);
    /// assert_eq!(true.then_some(0), Some(0));
    /// ```
    ///
    /// ```
    /// let mut a = 0;
    /// let mut function_with_side_effects = || { a += 1; };
    ///
    /// true.then_some(function_with_side_effects());
    /// false.then_some(function_with_side_effects());
    ///
    /// // `a` is incremented twice because the value passed to `then_some` is
    /// // evaluated eagerly.
    /// assert_eq!(a, 2);
    /// ```
    #[stable(feature = "bool_to_option", since = "1.62.0")]
    #[rustc_const_unstable(feature = "const_bool_to_option", issue = "91917")]
    #[inline]
    pub const fn then_some<T>(self, t: T) -> Option<T>
    where
        T: ~const Destruct,
    {
        if self { Some(t) } else { None }
    }

    /// Returns `Some(f())` if the `bool` is [`true`](../std/keyword.true.html),
    /// or `None` otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!(false.then(|| 0), None);
    /// assert_eq!(true.then(|| 0), Some(0));
    /// ```
    ///
    /// ```
    /// let mut a = 0;
    ///
    /// true.then(|| { a += 1; });
    /// false.then(|| { a += 1; });
    ///
    /// // `a` is incremented once because the closure is evaluated lazily by
    /// // `then`.
    /// assert_eq!(a, 1);
    /// ```
    #[stable(feature = "lazy_bool_to_option", since = "1.50.0")]
    #[rustc_const_unstable(feature = "const_bool_to_option", issue = "91917")]
    #[inline]
    pub const fn then<T, F>(self, f: F) -> Option<T>
    where
        F: ~const FnOnce() -> T,
        F: ~const Destruct,
    {
        if self { Some(f()) } else { None }
    }
}
