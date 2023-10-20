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
//! - Lines that are indented with more or less spaces than the first line
//! - Lines starting with `//`, `#[`, `)`, `]`, `}` if the comment has the same indentation as
//!   the first line
//!
//! If a line ends with an opening bracket, the line is ignored and the next line will have
//! its extra indentation ignored.

use std::{fmt::Display, path::Path};

use crate::walk::{filter_dirs, walk};

fn indentation(line: &str) -> usize {
    line.find(|c| c != ' ').unwrap_or(0)
}

fn is_close_bracket(c: char) -> bool {
    matches!(c, ')' | ']' | '}')
}

// Don't let tidy check this here :D
const START_MARKER: &str = concat!("tidy-alphabetical", "-start");
const END_MARKER: &str = "tidy-alphabetical-end";

fn check_section<'a>(
    file: impl Display,
    lines: impl Iterator<Item = (usize, &'a str)>,
    bad: &mut bool,
) {
    let mut prev_line = String::new();
    let mut first_indent = None;
    let mut in_split_line = None;

    for (line_idx, line) in lines {
        if line.contains(START_MARKER) {
            tidy_error!(bad, "{file}:{} found `{START_MARKER}` expecting `{END_MARKER}`", line_idx)
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
            || trimmed_line.starts_with("#[")
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
            tidy_error!(bad, "{file}:{}: line not in alphabetical order", line_idx + 1,);
        }

        prev_line = line;
    }

    tidy_error!(bad, "{file}: reached end of file expecting `{END_MARKER}`")
}

pub fn check(path: &Path, bad: &mut bool) {
    walk(path, |path, _is_dir| filter_dirs(path), &mut |entry, contents| {
        let file = &entry.path().display();

        let mut lines = contents.lines().enumerate();
        while let Some((_, line)) = lines.next() {
            if line.contains(START_MARKER) {
                check_section(file, &mut lines, bad);
            }
        }
    });
}
