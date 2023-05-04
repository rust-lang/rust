//! Missing batteries for standard libraries.

#![warn(rust_2018_idioms, unused_lifetimes, semicolon_in_expressions_from_macros)]

use std::io as sio;
use std::process::Command;
use std::{cmp::Ordering, ops, time::Instant};

mod macros;
pub mod hash;
pub mod process;
pub mod panic_context;
pub mod non_empty_vec;
pub mod rand;

pub use always_assert::{always, never};

#[inline(always)]
pub fn is_ci() -> bool {
    option_env!("CI").is_some()
}

#[must_use]
pub fn timeit(label: &'static str) -> impl Drop {
    let start = Instant::now();
    defer(move || eprintln!("{}: {:.2?}", label, start.elapsed()))
}

/// Prints backtrace to stderr, useful for debugging.
pub fn print_backtrace() {
    #[cfg(feature = "backtrace")]
    eprintln!("{:?}", backtrace::Backtrace::new());

    #[cfg(not(feature = "backtrace"))]
    eprintln!(
        r#"Enable the backtrace feature.
Uncomment `default = [ "backtrace" ]` in `crates/stdx/Cargo.toml`.
"#
    );
}

pub fn to_lower_snake_case(s: &str) -> String {
    to_snake_case(s, char::to_lowercase)
}
pub fn to_upper_snake_case(s: &str) -> String {
    to_snake_case(s, char::to_uppercase)
}

// Code partially taken from rust/compiler/rustc_lint/src/nonstandard_style.rs
// commit: 9626f2b
fn to_snake_case<F, I>(mut s: &str, change_case: F) -> String
where
    F: Fn(char) -> I,
    I: Iterator<Item = char>,
{
    let mut words = vec![];

    // Preserve leading underscores
    s = s.trim_start_matches(|c: char| {
        if c == '_' {
            words.push(String::new());
            true
        } else {
            false
        }
    });

    for s in s.split('_') {
        let mut last_upper = false;
        let mut buf = String::new();

        if s.is_empty() {
            continue;
        }

        for ch in s.chars() {
            if !buf.is_empty() && buf != "'" && ch.is_uppercase() && !last_upper {
                words.push(buf);
                buf = String::new();
            }

            last_upper = ch.is_uppercase();
            buf.extend(change_case(ch));
        }

        words.push(buf);
    }

    words.join("_")
}

pub fn replace(buf: &mut String, from: char, to: &str) {
    if !buf.contains(from) {
        return;
    }
    // FIXME: do this in place.
    *buf = buf.replace(from, to);
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
    text.split_inclusive('\n')
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

pub fn equal_range_by<T, F>(slice: &[T], mut key: F) -> ops::Range<usize>
where
    F: FnMut(&T) -> Ordering,
{
    let start = slice.partition_point(|it| key(it) == Ordering::Less);
    let len = slice[start..].partition_point(|it| key(it) == Ordering::Equal);
    start..start + len
}

#[must_use]
pub fn defer<F: FnOnce()>(f: F) -> impl Drop {
    struct D<F: FnOnce()>(Option<F>);
    impl<F: FnOnce()> Drop for D<F> {
        fn drop(&mut self) {
            if let Some(f) = self.0.take() {
                f();
            }
        }
    }
    D(Some(f))
}

/// A [`std::process::Child`] wrapper that will kill the child on drop.
#[cfg_attr(not(target_arch = "wasm32"), repr(transparent))]
#[derive(Debug)]
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
    pub fn spawn(mut command: Command) -> sio::Result<Self> {
        command.spawn().map(Self)
    }

    pub fn into_inner(self) -> std::process::Child {
        if cfg!(target_arch = "wasm32") {
            panic!("no processes on wasm");
        }
        // SAFETY: repr transparent, except on WASM
        unsafe { std::mem::transmute::<JodChild, std::process::Child>(self) }
    }
}

// feature: iter_order_by
// Iterator::eq_by
pub fn iter_eq_by<I, I2, F>(this: I2, other: I, mut eq: F) -> bool
where
    I: IntoIterator,
    I2: IntoIterator,
    F: FnMut(I2::Item, I::Item) -> bool,
{
    let mut other = other.into_iter();
    let mut this = this.into_iter();

    loop {
        let x = match this.next() {
            None => return other.next().is_none(),
            Some(val) => val,
        };

        let y = match other.next() {
            None => return false,
            Some(val) => val,
        };

        if !eq(x, y) {
            return false;
        }
    }
}

/// Returns all final segments of the argument, longest first.
pub fn slice_tails<T>(this: &[T]) -> impl Iterator<Item = &[T]> {
    (0..this.len()).map(|i| &this[i..])
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
