//! impl bool {}

#[lang = "bool"]
impl bool {
    /// Returns `Some(t)` if the `bool` is [`true`](../std/keyword.true.html),
    /// or `None` otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(bool_to_option)]
    ///
    /// assert_eq!(false.then_some(0), None);
    /// assert_eq!(true.then_some(0), Some(0));
    /// ```
    #[unstable(feature = "bool_to_option", issue = "80967")]
    #[rustc_const_unstable(feature = "const_bool_to_option", issue = "91917")]
    #[inline]
    pub const fn then_some<T>(self, t: T) -> Option<T>
    where
        T: ~const Drop,
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
    #[stable(feature = "lazy_bool_to_option", since = "1.50.0")]
    #[rustc_const_unstable(feature = "const_bool_to_option", issue = "91917")]
    #[inline]
    pub const fn then<T, F>(self, f: F) -> Option<T>
    where
        F: ~const FnOnce() -> T,
        F: ~const Drop,
    {
        if self { Some(f()) } else { None }
    }
}
