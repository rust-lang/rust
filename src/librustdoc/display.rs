//! Various utilities for working with [`fmt::Display`] implementations.

use std::fmt::{self, Display, Formatter};

pub(crate) trait Joined: IntoIterator {
    /// Takes an iterator over elements that implement [`Display`], and format them into `f`, separated by `sep`.
    ///
    /// This is similar to [`Itertools::format`](itertools::Itertools::format), but instead of returning an implementation of `Display`,
    /// it formats directly into a [`Formatter`].
    ///
    /// The performance of `joined` is slightly better than `format`, since it doesn't need to use a `Cell` to keep track of whether [`fmt`](Display::fmt)
    /// was already called (`joined`'s API doesn't allow it be called more than once).
    fn joined(self, sep: &str, f: &mut Formatter<'_>) -> fmt::Result;
}

impl<I, T> Joined for I
where
    I: IntoIterator<Item = T>,
    T: Display,
{
    fn joined(self, sep: &str, f: &mut Formatter<'_>) -> fmt::Result {
        let mut iter = self.into_iter();
        let Some(first) = iter.next() else { return Ok(()) };
        first.fmt(f)?;
        for item in iter {
            f.write_str(sep)?;
            item.fmt(f)?;
        }
        Ok(())
    }
}

pub(crate) trait MaybeDisplay {
    /// For a given `Option<T: Display>`, returns a `Display` implementation that will display `t` if `Some(t)`, or nothing if `None`.
    fn maybe_display(self) -> impl Display;
}

impl<T: Display> MaybeDisplay for Option<T> {
    fn maybe_display(self) -> impl Display {
        fmt::from_fn(move |f| {
            if let Some(t) = self.as_ref() {
                t.fmt(f)?;
            }
            Ok(())
        })
    }
}
