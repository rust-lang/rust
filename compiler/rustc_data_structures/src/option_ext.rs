/// Extension for [`Option`] that adds handy methods missing from it.
pub trait OptionExt {
    type T;

    fn is_none_or(self, f: impl FnOnce(Self::T) -> bool) -> bool;
}

impl<T> OptionExt for Option<T> {
    type T = T;

    /// Returns `true` is `self` is `None` or the value inside matches a predicate `f`.
    fn is_none_or(self, f: impl FnOnce(T) -> bool) -> bool {
        match self {
            None => true,
            Some(x) => f(x),
        }
    }
}
