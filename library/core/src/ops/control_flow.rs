use crate::ops::Try;

/// Used to make try_fold closures more like normal loops
#[unstable(feature = "control_flow_enum", reason = "new API", issue = "75744")]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ControlFlow<B, C = ()> {
    /// Continue in the loop, using the given value for the next iteration
    Continue(C),
    /// Exit the loop, yielding the given value
    Break(B),
}

#[unstable(feature = "control_flow_enum", reason = "new API", issue = "75744")]
impl<B, C> Try for ControlFlow<B, C> {
    type Ok = C;
    type Error = B;
    #[inline]
    fn into_result(self) -> Result<Self::Ok, Self::Error> {
        match self {
            ControlFlow::Continue(y) => Ok(y),
            ControlFlow::Break(x) => Err(x),
        }
    }
    #[inline]
    fn from_error(v: Self::Error) -> Self {
        ControlFlow::Break(v)
    }
    #[inline]
    fn from_ok(v: Self::Ok) -> Self {
        ControlFlow::Continue(v)
    }
}

impl<B, C> ControlFlow<B, C> {
    /// Returns `true` if this is a `Break` variant.
    #[inline]
    #[unstable(feature = "control_flow_enum", reason = "new API", issue = "75744")]
    pub fn is_break(&self) -> bool {
        matches!(*self, ControlFlow::Break(_))
    }

    /// Returns `true` if this is a `Continue` variant.
    #[inline]
    #[unstable(feature = "control_flow_enum", reason = "new API", issue = "75744")]
    pub fn is_continue(&self) -> bool {
        matches!(*self, ControlFlow::Continue(_))
    }

    /// Converts the `ControlFlow` into an `Option` which is `Some` if the
    /// `ControlFlow` was `Break` and `None` otherwise.
    #[inline]
    #[unstable(feature = "control_flow_enum", reason = "new API", issue = "75744")]
    pub fn break_value(self) -> Option<B> {
        match self {
            ControlFlow::Continue(..) => None,
            ControlFlow::Break(x) => Some(x),
        }
    }

    /// Maps `ControlFlow<B, C>` to `ControlFlow<T, C>` by applying a function
    /// to the break value in case it exists.
    #[inline]
    #[unstable(feature = "control_flow_enum", reason = "new API", issue = "75744")]
    pub fn map_break<T, F>(self, f: F) -> ControlFlow<T, C>
    where
        F: FnOnce(B) -> T,
    {
        match self {
            ControlFlow::Continue(x) => ControlFlow::Continue(x),
            ControlFlow::Break(x) => ControlFlow::Break(f(x)),
        }
    }
}

impl<R: Try> ControlFlow<R, R::Ok> {
    /// Create a `ControlFlow` from any type implementing `Try`.
    #[unstable(feature = "control_flow_enum", reason = "new API", issue = "75744")]
    #[inline]
    pub fn from_try(r: R) -> Self {
        match Try::into_result(r) {
            Ok(v) => ControlFlow::Continue(v),
            Err(v) => ControlFlow::Break(Try::from_error(v)),
        }
    }

    /// Convert a `ControlFlow` into any type implementing `Try`;
    #[unstable(feature = "control_flow_enum", reason = "new API", issue = "75744")]
    #[inline]
    pub fn into_try(self) -> R {
        match self {
            ControlFlow::Continue(v) => Try::from_ok(v),
            ControlFlow::Break(v) => v,
        }
    }
}

impl<B> ControlFlow<B, ()> {
    /// It's frequently the case that there's no value needed with `Continue`,
    /// so this provides a way to avoid typing `(())`, if you prefer it.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(control_flow_enum)]
    /// use std::ops::ControlFlow;
    ///
    /// let mut partial_sum = 0;
    /// let last_used = (1..10).chain(20..25).try_for_each(|x| {
    ///     partial_sum += x;
    ///     if partial_sum > 100 { ControlFlow::Break(x) }
    ///     else { ControlFlow::CONTINUE }
    /// });
    /// assert_eq!(last_used.break_value(), Some(22));
    /// ```
    #[unstable(feature = "control_flow_enum", reason = "new API", issue = "75744")]
    pub const CONTINUE: Self = ControlFlow::Continue(());
}

impl<C> ControlFlow<(), C> {
    /// APIs like `try_for_each` don't need values with `Break`,
    /// so this provides a way to avoid typing `(())`, if you prefer it.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(control_flow_enum)]
    /// use std::ops::ControlFlow;
    ///
    /// let mut partial_sum = 0;
    /// (1..10).chain(20..25).try_for_each(|x| {
    ///     if partial_sum > 100 { ControlFlow::BREAK }
    ///     else { partial_sum += x; ControlFlow::CONTINUE }
    /// });
    /// assert_eq!(partial_sum, 108);
    /// ```
    #[unstable(feature = "control_flow_enum", reason = "new API", issue = "75744")]
    pub const BREAK: Self = ControlFlow::Break(());
}
