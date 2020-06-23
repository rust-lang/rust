//! Missing batteries for standard libraries.
use std::{cell::Cell, fmt, time::Instant};

#[inline(always)]
pub fn is_ci() -> bool {
    option_env!("CI").is_some()
}

#[macro_export]
macro_rules! eprintln {
    ($($tt:tt)*) => {{
        if $crate::is_ci() {
            panic!("Forgot to remove debug-print?")
        }
        std::eprintln!($($tt)*)
    }}
}

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
pub fn timeit(label: &'static str) -> impl Drop {
    struct Guard {
        label: &'static str,
        start: Instant,
    }

    impl Drop for Guard {
        fn drop(&mut self) {
            eprintln!("{}: {:?}", self.label, self.start.elapsed())
        }
    }

    Guard { label, start: Instant::now() }
}

pub fn to_lower_snake_case(s: &str) -> String {
    let mut buf = String::with_capacity(s.len());
    let mut prev = false;
    for c in s.chars() {
        if c.is_ascii_uppercase() && prev {
            buf.push('_')
        }
        prev = true;

        buf.push(c.to_ascii_lowercase());
    }
    buf
}

pub fn replace(buf: &mut String, from: char, to: &str) {
    if !buf.contains(from) {
        return;
    }
    // FIXME: do this in place.
    *buf = buf.replace(from, to)
}

pub fn split_delim(haystack: &str, delim: char) -> Option<(&str, &str)> {
    let idx = haystack.find(delim)?;
    Some((&haystack[..idx], &haystack[idx + delim.len_utf8()..]))
}

pub fn trim_indent(mut text: &str) -> String {
    if text.starts_with('\n') {
        text = &text[1..];
    }
    let indent = text
        .lines()
        .filter(|it| !it.trim().is_empty())
        .map(|it| it.len() - it.trim_start().len())
        .min()
        .unwrap_or(0);
    lines_with_ends(text)
        .map(
            |line| {
                if line.len() <= indent {
                    line.trim_start_matches(' ')
                } else {
                    &line[indent..]
                }
            },
        )
        .collect()
}

pub fn lines_with_ends(text: &str) -> LinesWithEnds {
    LinesWithEnds { text }
}

pub struct LinesWithEnds<'a> {
    text: &'a str,
}

impl<'a> Iterator for LinesWithEnds<'a> {
    type Item = &'a str;
    fn next(&mut self) -> Option<&'a str> {
        if self.text.is_empty() {
            return None;
        }
        let idx = self.text.find('\n').map_or(self.text.len(), |it| it + 1);
        let (res, next) = self.text.split_at(idx);
        self.text = next;
        Some(res)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trim_indent() {
        assert_eq!(trim_indent(""), "");
        assert_eq!(
            trim_indent(
                "
            hello
            world
"
            ),
            "hello\nworld\n"
        );
        assert_eq!(
            trim_indent(
                "
            hello
            world"
            ),
            "hello\nworld"
        );
        assert_eq!(trim_indent("    hello\n    world\n"), "hello\nworld\n");
        assert_eq!(
            trim_indent(
                "
            fn main() {
                return 92;
            }
        "
            ),
            "fn main() {\n    return 92;\n}\n"
        );
    }
}
