//! impl bool {}

#[lang = "bool"]
impl bool {
    /// Returns `Some(t)` if the `bool` is `true`, or `None` otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(bool_to_option)]
    ///
    /// assert_eq!(false.to_option(0), None);
    /// assert_eq!(true.to_option(0), Some(0));
    /// ```
    #[unstable(feature = "bool_to_option", issue = "64260")]
    #[inline]
    pub fn to_option<T>(self, t: T) -> Option<T> {
        if self {
            Some(t)
        } else {
            None
        }
    }

    /// Returns `Some(f())` if the `bool` is `true`, or `None` otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(bool_to_option)]
    ///
    /// assert_eq!(false.to_option_with(|| 0), None);
    /// assert_eq!(true.to_option_with(|| 0), Some(0));
    /// ```
    #[unstable(feature = "bool_to_option", issue = "64260")]
    #[inline]
    pub fn to_option_with<T, F: FnOnce() -> T>(self, f: F) -> Option<T> {
        if self {
            Some(f())
        } else {
            None
        }
    }
}
