//! Various utilities for working with [`fmt::Display`] implementations.

use std::fmt::{self, Display, Formatter, FormattingOptions};

pub(crate) trait Joined: IntoIterator {
    /// Takes an iterator over elements that implement [`Display`], and format them into `f`, separated by `sep`.
    ///
    /// This is similar to [`Itertools::format`](itertools::Itertools::format), but instead of returning an implementation of `Display`,
    /// it formats directly into a [`Formatter`].
    ///
    /// The performance of `joined` is slightly better than `format`, since it doesn't need to use a `Cell` to keep track of whether [`fmt`](Display::fmt)
    /// was already called (`joined`'s API doesn't allow it be called more than once).
    fn joined(self, sep: impl Display, f: &mut Formatter<'_>) -> fmt::Result;
}

impl<I, T> Joined for I
where
    I: IntoIterator<Item = T>,
    T: Display,
{
    fn joined(self, sep: impl Display, f: &mut Formatter<'_>) -> fmt::Result {
        let mut iter = self.into_iter();
        let Some(first) = iter.next() else { return Ok(()) };
        first.fmt(f)?;
        for item in iter {
            sep.fmt(f)?;
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

#[derive(Clone, Copy)]
pub(crate) struct Wrapped<T> {
    prefix: T,
    suffix: T,
}

pub(crate) enum AngleBracket {
    Open,
    Close,
}

impl Display for AngleBracket {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(match (self, f.alternate()) {
            (Self::Open, true) => "<",
            (Self::Open, false) => "&lt;",
            (Self::Close, true) => ">",
            (Self::Close, false) => "&gt;",
        })
    }
}

impl Wrapped<AngleBracket> {
    pub(crate) fn with_angle_brackets() -> Self {
        Self { prefix: AngleBracket::Open, suffix: AngleBracket::Close }
    }
}

impl Wrapped<char> {
    pub(crate) fn with_parens() -> Self {
        Self { prefix: '(', suffix: ')' }
    }

    pub(crate) fn with_square_brackets() -> Self {
        Self { prefix: '[', suffix: ']' }
    }
}

impl<T: Display> Wrapped<T> {
    pub(crate) fn with(prefix: T, suffix: T) -> Self {
        Self { prefix, suffix }
    }

    pub(crate) fn when(self, if_: bool) -> Wrapped<impl Display> {
        Wrapped {
            prefix: if_.then_some(self.prefix).maybe_display(),
            suffix: if_.then_some(self.suffix).maybe_display(),
        }
    }

    pub(crate) fn wrap_fn(
        self,
        content: impl Fn(&mut Formatter<'_>) -> fmt::Result,
    ) -> impl Display {
        fmt::from_fn(move |f| {
            self.prefix.fmt(f)?;
            content(f)?;
            self.suffix.fmt(f)
        })
    }

    pub(crate) fn wrap<C: Display>(self, content: C) -> impl Display {
        self.wrap_fn(move |f| content.fmt(f))
    }
}

#[derive(Clone, Copy)]
pub(crate) struct WithOpts {
    opts: FormattingOptions,
}

impl WithOpts {
    pub(crate) fn from(f: &Formatter<'_>) -> Self {
        Self { opts: f.options() }
    }

    pub(crate) fn display(self, t: impl Display) -> impl Display {
        fmt::from_fn(move |f| {
            let mut f = f.with_options(self.opts);
            t.fmt(&mut f)
        })
    }
}
