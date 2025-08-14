//! Checks that a list of items is in alphabetical order
//!
//! Use the following marker in the code:
//! ```rust
//! // tidy-alphabetical-start
//! fn aaa() {}
//! fn eee() {}
//! fn z() {}
//! // tidy-alphabetical-end
//! ```
//!
//! The following lines are ignored:
//! - Empty lines
//! - Lines that are indented with more or less spaces than the first line
//! - Lines starting with `//`, `#` (except those starting with `#!`), `)`, `]`, `}` if the comment
//!   has the same indentation as the first line
//! - Lines starting with a closing delimiter (`)`, `[`, `}`) are ignored.
//!
//! If a line ends with an opening delimiter, we effectively join the following line to it before
//! checking it. E.g. `foo(\nbar)` is treated like `foo(bar)`.

use std::cmp::Ordering;
use std::fmt::Display;
use std::iter::Peekable;
use std::path::Path;

use crate::walk::{filter_dirs, walk};

#[cfg(test)]
mod tests;

fn indentation(line: &str) -> usize {
    line.find(|c| c != ' ').unwrap_or(0)
}

fn is_close_bracket(c: char) -> bool {
    matches!(c, ')' | ']' | '}')
}

const START_MARKER: &str = "tidy-alphabetical-start";
const END_MARKER: &str = "tidy-alphabetical-end";

fn check_section<'a>(
    file: impl Display,
    lines: impl Iterator<Item = (usize, &'a str)>,
    err: &mut dyn FnMut(&str) -> std::io::Result<()>,
    bad: &mut bool,
) {
    let mut prev_line = String::new();
    let mut first_indent = None;
    let mut in_split_line = None;

    for (idx, line) in lines {
        if line.is_empty() {
            continue;
        }

        if line.contains(START_MARKER) {
            tidy_error_ext!(
                err,
                bad,
                "{file}:{} found `{START_MARKER}` expecting `{END_MARKER}`",
                idx + 1
            );
            return;
        }

        if line.contains(END_MARKER) {
            return;
        }

        let indent = first_indent.unwrap_or_else(|| {
            let indent = indentation(line);
            first_indent = Some(indent);
            indent
        });

        let line = if let Some(prev_split_line) = in_split_line {
            // Join the split lines.
            in_split_line = None;
            format!("{prev_split_line}{}", line.trim_start())
        } else {
            line.to_string()
        };

        if indentation(&line) != indent {
            continue;
        }

        let trimmed_line = line.trim_start_matches(' ');

        if trimmed_line.starts_with("//")
            || (trimmed_line.starts_with('#') && !trimmed_line.starts_with("#!"))
            || trimmed_line.starts_with(is_close_bracket)
        {
            continue;
        }

        if line.trim_end().ends_with('(') {
            in_split_line = Some(line);
            continue;
        }

        let prev_line_trimmed_lowercase = prev_line.trim_start_matches(' ');

        if version_sort(trimmed_line, prev_line_trimmed_lowercase).is_lt() {
            tidy_error_ext!(err, bad, "{file}:{}: line not in alphabetical order", idx + 1);
        }

        prev_line = line;
    }

    tidy_error_ext!(err, bad, "{file}: reached end of file expecting `{END_MARKER}`")
}

fn check_lines<'a>(
    file: &impl Display,
    mut lines: impl Iterator<Item = (usize, &'a str)>,
    err: &mut dyn FnMut(&str) -> std::io::Result<()>,
    bad: &mut bool,
) {
    while let Some((idx, line)) = lines.next() {
        if line.contains(END_MARKER) {
            tidy_error_ext!(
                err,
                bad,
                "{file}:{} found `{END_MARKER}` expecting `{START_MARKER}`",
                idx + 1
            )
        }

        if line.contains(START_MARKER) {
            check_section(file, &mut lines, err, bad);
        }
    }
}

pub fn check(path: &Path, bad: &mut bool) {
    let skip =
        |path: &_, _is_dir| filter_dirs(path) || path.ends_with("tidy/src/alphabetical/tests.rs");

    walk(path, skip, &mut |entry, contents| {
        let file = &entry.path().display();
        let lines = contents.lines().enumerate();
        check_lines(file, lines, &mut crate::tidy_error, bad)
    });
}

fn consume_numeric_prefix<I: Iterator<Item = char>>(it: &mut Peekable<I>) -> String {
    let mut result = String::new();

    while let Some(&c) = it.peek() {
        if !c.is_numeric() {
            break;
        }

        result.push(c);
        it.next();
    }

    result
}

// A sorting function that is case-sensitive, and sorts sequences of digits by their numeric value,
// so that `9` sorts before `12`.
fn version_sort(a: &str, b: &str) -> Ordering {
    let mut it1 = a.chars().peekable();
    let mut it2 = b.chars().peekable();

    while let (Some(x), Some(y)) = (it1.peek(), it2.peek()) {
        match (x.is_numeric(), y.is_numeric()) {
            (true, true) => {
                let num1: String = consume_numeric_prefix(it1.by_ref());
                let num2: String = consume_numeric_prefix(it2.by_ref());

                let int1: u64 = num1.parse().unwrap();
                let int2: u64 = num2.parse().unwrap();

                // Compare strings when the numeric value is equal to handle "00" versus "0".
                match int1.cmp(&int2).then_with(|| num1.cmp(&num2)) {
                    Ordering::Equal => continue,
                    different => return different,
                }
            }
            (false, false) => match x.cmp(y) {
                Ordering::Equal => {
                    it1.next();
                    it2.next();
                    continue;
                }
                different => return different,
            },
            (false, true) | (true, false) => {
                return x.cmp(y);
            }
        }
    }

    it1.next().cmp(&it2.next())
}
