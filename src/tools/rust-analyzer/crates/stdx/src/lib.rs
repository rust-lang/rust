//! Missing batteries for standard libraries.

use std::io as sio;
use std::process::Command;
use std::{cmp::Ordering, ops, time::Instant};

mod macros;

pub mod anymap;
pub mod assert;
pub mod non_empty_vec;
pub mod panic_context;
pub mod process;
pub mod rand;
pub mod thread;
pub mod variance;

pub use itertools;

#[inline(always)]
pub const fn is_ci() -> bool {
    option_env!("CI").is_some()
}

pub fn hash_once<Hasher: std::hash::Hasher + Default>(thing: impl std::hash::Hash) -> u64 {
    std::hash::BuildHasher::hash_one(&std::hash::BuildHasherDefault::<Hasher>::default(), thing)
}

#[must_use]
#[expect(clippy::print_stderr, reason = "only visible to developers")]
pub fn timeit(label: &'static str) -> impl Drop {
    let start = Instant::now();
    defer(move || eprintln!("{}: {:.2}", label, start.elapsed().as_nanos()))
}

/// Prints backtrace to stderr, useful for debugging.
#[expect(clippy::print_stderr, reason = "only visible to developers")]
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

pub trait TupleExt {
    type Head;
    type Tail;
    fn head(self) -> Self::Head;
    fn tail(self) -> Self::Tail;
}

impl<T, U> TupleExt for (T, U) {
    type Head = T;
    type Tail = U;
    fn head(self) -> Self::Head {
        self.0
    }
    fn tail(self) -> Self::Tail {
        self.1
    }
}

impl<T, U, V> TupleExt for (T, U, V) {
    type Head = T;
    type Tail = V;
    fn head(self) -> Self::Head {
        self.0
    }
    fn tail(self) -> Self::Tail {
        self.2
    }
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

// Taken from rustc.
#[must_use]
pub fn to_camel_case(ident: &str) -> String {
    ident
        .trim_matches('_')
        .split('_')
        .filter(|component| !component.is_empty())
        .map(|component| {
            let mut camel_cased_component = String::with_capacity(component.len());

            let mut new_word = true;
            let mut prev_is_lower_case = true;

            for c in component.chars() {
                // Preserve the case if an uppercase letter follows a lowercase letter, so that
                // `camelCase` is converted to `CamelCase`.
                if prev_is_lower_case && c.is_uppercase() {
                    new_word = true;
                }

                if new_word {
                    camel_cased_component.extend(c.to_uppercase());
                } else {
                    camel_cased_component.extend(c.to_lowercase());
                }

                prev_is_lower_case = c.is_lowercase();
                new_word = false;
            }

            camel_cased_component
        })
        .fold((String::new(), None), |(mut acc, prev): (_, Option<String>), next| {
            // separate two components with an underscore if their boundary cannot
            // be distinguished using an uppercase/lowercase case distinction
            let join = prev
                .and_then(|prev| {
                    let f = next.chars().next()?;
                    let l = prev.chars().last()?;
                    Some(!char_has_case(l) && !char_has_case(f))
                })
                .unwrap_or(false);
            acc.push_str(if join { "_" } else { "" });
            acc.push_str(&next);
            (acc, Some(next))
        })
        .0
}

// Taken from rustc.
#[must_use]
pub const fn char_has_case(c: char) -> bool {
    c.is_lowercase() || c.is_uppercase()
}

#[must_use]
pub fn is_upper_snake_case(s: &str) -> bool {
    s.chars().all(|c| c.is_uppercase() || c == '_' || c.is_numeric())
}

pub fn replace(buf: &mut String, from: char, to: &str) {
    if !buf.contains(from) {
        return;
    }
    // FIXME: do this in place.
    *buf = buf.replace(from, to);
}

#[must_use]
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
                if line.len() <= indent { line.trim_start_matches(' ') } else { &line[indent..] }
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
        _ = self.0.kill();
        _ = self.0.wait();
    }
}

impl JodChild {
    pub fn spawn(mut command: Command) -> sio::Result<Self> {
        command.spawn().map(Self)
    }

    #[must_use]
    #[cfg(not(target_arch = "wasm32"))]
    pub fn into_inner(self) -> std::process::Child {
        // SAFETY: repr transparent, except on WASM
        unsafe { std::mem::transmute::<Self, std::process::Child>(self) }
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
