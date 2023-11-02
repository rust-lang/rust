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

use std::fmt::Display;
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
            || (trimmed_line.starts_with("#") && !trimmed_line.starts_with("#!"))
            || trimmed_line.starts_with(is_close_bracket)
        {
            continue;
        }

        if line.trim_end().ends_with('(') {
            in_split_line = Some(line);
            continue;
        }

        let prev_line_trimmed_lowercase = prev_line.trim_start_matches(' ').to_lowercase();

        if trimmed_line.to_lowercase() < prev_line_trimmed_lowercase {
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
