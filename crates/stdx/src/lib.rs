//! Missing batteries for standard libraries.

use std::{cell::Cell, fmt};

/// Appends formatted string to a `String`.
#[macro_export]
macro_rules! format_to {
    ($buf:expr) => ();
    ($buf:expr, $lit:literal $($arg:tt)*) => {
        { use ::std::fmt::Write as _; let _ = ::std::write!($buf, $lit $($arg)*); }
    };
}

pub trait SepBy: Sized {
    /// Returns an `impl fmt::Display`, which joins elements via a separator.
    fn sep_by<'a>(self, sep: &'a str) -> SepByBuilder<'a, Self>;
}

impl<I> SepBy for I
where
    I: Iterator,
    I::Item: fmt::Display,
{
    fn sep_by<'a>(self, sep: &'a str) -> SepByBuilder<'a, Self> {
        SepByBuilder::new(sep, self)
    }
}

pub struct SepByBuilder<'a, I> {
    sep: &'a str,
    prefix: &'a str,
    suffix: &'a str,
    iter: Cell<Option<I>>,
}

impl<'a, I> SepByBuilder<'a, I> {
    fn new(sep: &'a str, iter: I) -> SepByBuilder<'a, I> {
        SepByBuilder { sep, prefix: "", suffix: "", iter: Cell::new(Some(iter)) }
    }

    pub fn prefix(mut self, prefix: &'a str) -> Self {
        self.prefix = prefix;
        self
    }

    pub fn suffix(mut self, suffix: &'a str) -> Self {
        self.suffix = suffix;
        self
    }

    /// Set both suffix and prefix.
    pub fn surround_with(self, prefix: &'a str, suffix: &'a str) -> Self {
        self.prefix(prefix).suffix(suffix)
    }
}

impl<I> fmt::Display for SepByBuilder<'_, I>
where
    I: Iterator,
    I::Item: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.prefix)?;
        let mut first = true;
        for item in self.iter.take().unwrap() {
            if !first {
                f.write_str(self.sep)?;
            }
            first = false;
            fmt::Display::fmt(&item, f)?;
        }
        f.write_str(self.suffix)?;
        Ok(())
    }
}
