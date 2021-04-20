//! Missing batteries for standard libraries.
use std::{cmp::Ordering, ops, time::Instant};

mod macros;
pub mod process;
pub mod panic_context;

pub use always_assert::{always, never};

#[inline(always)]
pub fn is_ci() -> bool {
    option_env!("CI").is_some()
}

#[must_use]
pub fn timeit(label: &'static str) -> impl Drop {
    struct Guard {
        label: &'static str,
        start: Instant,
    }

    impl Drop for Guard {
        fn drop(&mut self) {
            eprintln!("{}: {:.2?}", self.label, self.start.elapsed())
        }
    }

    Guard { label, start: Instant::now() }
}

/// Prints backtrace to stderr, useful for debugging.
#[cfg(feature = "backtrace")]
pub fn print_backtrace() {
    let bt = backtrace::Backtrace::new();
    eprintln!("{:?}", bt);
}
#[cfg(not(feature = "backtrace"))]
pub fn print_backtrace() {
    eprintln!(
        r#"Enable the backtrace feature.
Uncomment `default = [ "backtrace" ]` in `crates/stdx/Cargo.toml`.
"#
    );
}

pub fn to_lower_snake_case(s: &str) -> String {
    to_snake_case(s, char::to_ascii_lowercase)
}
pub fn to_upper_snake_case(s: &str) -> String {
    to_snake_case(s, char::to_ascii_uppercase)
}
fn to_snake_case<F: Fn(&char) -> char>(s: &str, change_case: F) -> String {
    let mut buf = String::with_capacity(s.len());
    let mut prev = false;
    for c in s.chars() {
        // `&& prev` is required to not insert `_` before the first symbol.
        if c.is_ascii_uppercase() && prev {
            // This check is required to not translate `Weird_Case` into `weird__case`.
            if !buf.ends_with('_') {
                buf.push('_')
            }
        }
        prev = true;

        buf.push(change_case(&c));
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

// https://github.com/rust-lang/rust/issues/74773
pub fn split_once(haystack: &str, delim: char) -> Option<(&str, &str)> {
    let mut split = haystack.splitn(2, delim);
    let prefix = split.next()?;
    let suffix = split.next()?;
    Some((prefix, suffix))
}
pub fn rsplit_once(haystack: &str, delim: char) -> Option<(&str, &str)> {
    let mut split = haystack.rsplitn(2, delim);
    let suffix = split.next()?;
    let prefix = split.next()?;
    Some((prefix, suffix))
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

/// Returns `idx` such that:
///
/// ```text
///     ∀ x in slice[..idx]:  pred(x)
///  && ∀ x in slice[idx..]: !pred(x)
/// ```
///
/// https://github.com/rust-lang/rust/issues/73831
pub fn partition_point<T, P>(slice: &[T], mut pred: P) -> usize
where
    P: FnMut(&T) -> bool,
{
    let mut left = 0;
    let mut right = slice.len();

    while left != right {
        let mid = left + (right - left) / 2;
        // SAFETY:
        // When left < right, left <= mid < right.
        // Therefore left always increases and right always decreases,
        // and either of them is selected.
        // In both cases left <= right is satisfied.
        // Therefore if left < right in a step,
        // left <= right is satisfied in the next step.
        // Therefore as long as left != right, 0 <= left < right <= len is satisfied
        // and if this case 0 <= mid < len is satisfied too.
        let value = unsafe { slice.get_unchecked(mid) };
        if pred(value) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }

    left
}

pub fn equal_range_by<T, F>(slice: &[T], mut key: F) -> ops::Range<usize>
where
    F: FnMut(&T) -> Ordering,
{
    let start = partition_point(slice, |it| key(it) == Ordering::Less);
    let len = partition_point(&slice[start..], |it| key(it) == Ordering::Equal);
    start..start + len
}

pub fn defer<F: FnOnce()>(f: F) -> impl Drop {
    struct D<F: FnOnce()>(Option<F>);
    impl<F: FnOnce()> Drop for D<F> {
        fn drop(&mut self) {
            if let Some(f) = self.0.take() {
                f()
            }
        }
    }
    D(Some(f))
}

#[repr(transparent)]
pub struct JodChild(pub std::process::Child);

impl ops::Deref for JodChild {
    type Target = std::process::Child;
    fn deref(&self) -> &std::process::Child {
        &self.0
    }
}

impl ops::DerefMut for JodChild {
    fn deref_mut(&mut self) -> &mut std::process::Child {
        &mut self.0
    }
}

impl Drop for JodChild {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

impl JodChild {
    pub fn into_inner(self) -> std::process::Child {
        // SAFETY: repr transparent
        unsafe { std::mem::transmute::<JodChild, std::process::Child>(self) }
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
