//! impl bool {}

use crate::marker::Destruct;

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

    /// Flips the `bool` value turning false to true and true to false
    ///
    /// # Examples
    ///
    /// ```
    /// assert!(false.flip())
    /// asssert!(!true.flip())
    /// ```
    #[unstable(feature = "flip_bool", issue = "none", reason = "recently added")]
    #[inline]
    pub const fn flip(&self) -> Self {
        !self
    }

    /// Toggles the `bool` variable and mutates it
    #[unstable(feature = "toggle_bool", issue = "none", reason = "recently added")]
    #[inline]
    pub const fn toggle(&mut self) {
        *self = self.flip();
    }
}
