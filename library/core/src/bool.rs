//! impl bool {}

impl bool {
    /// Returns `Some(t)` if the `bool` is [`true`](../std/keyword.true.html),
    /// or `None` otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!(false.then_some(0), None);
    /// assert_eq!(true.then_some(0), Some(0));
    /// ```
    #[stable(feature = "bool_to_option", since = "1.62.0")]
    #[inline]
    pub fn then_some<T>(self, t: T) -> Option<T> {
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
    #[inline]
    pub fn then<T, F>(self, f: F) -> Option<T>
    where
        F: FnOnce() -> T,
    {
        if self { Some(f()) } else { None }
    }
}
