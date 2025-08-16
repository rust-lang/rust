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
//! Empty lines and lines starting (ignoring spaces) with `//` or `#` (except those starting with
//! `#!`) are considered comments are are sorted together with the next line (but do not affect
//! sorting).
//!
//! If the following lines have higher indentation we effectively join them with the current line
//! before comparing it. If the next line with the same indentation starts (ignoring spaces) with
//! a closing delimiter (`)`, `[`, `}`) it is joined as well.
//!
//! E.g.
//!
//! ```rust,ignore ilustrative example for sorting mentioning non-existent functions
//! foo(a,
//!     b);
//! bar(
//!   a,
//!   b
//! );
//! // are treated for sorting purposes as
//! foo(a, b);
//! bar(a, b);
//! ```

use std::cmp::Ordering;
use std::fs;
use std::io::{Seek, Write};
use std::iter::Peekable;
use std::ops::{Range, RangeBounds};
use std::path::Path;

use crate::diagnostics::{CheckId, RunningCheck, TidyCtx};
use crate::walk::{filter_dirs, walk};

#[cfg(test)]
mod tests;

fn indentation(line: &str) -> usize {
    line.find(|c| c != ' ').unwrap_or(0)
}

fn is_close_bracket(c: char) -> bool {
    matches!(c, ')' | ']' | '}')
}

fn is_empty_or_comment(line: &&str) -> bool {
    let trimmed_line = line.trim_start_matches(' ').trim_end_matches('\n');

    trimmed_line.is_empty()
        || trimmed_line.starts_with("//")
        || (trimmed_line.starts_with('#') && !trimmed_line.starts_with("#!"))
}

const START_MARKER: &str = "tidy-alphabetical-start";
const END_MARKER: &str = "tidy-alphabetical-end";

/// Given contents of a section that is enclosed between [`START_MARKER`] and [`END_MARKER`], sorts
/// them according to the rules described at the top of the module.
fn sort_section(section: &str) -> String {
    /// A sortable item
    struct Item {
        /// Full contents including comments and whitespace
        full: String,
        /// Trimmed contents for sorting
        trimmed: String,
    }

    let mut items = Vec::new();
    let mut lines = section.split_inclusive('\n').peekable();

    let end_comments = loop {
        let mut full = String::new();
        let mut trimmed = String::new();

        while let Some(comment) = lines.next_if(is_empty_or_comment) {
            full.push_str(comment);
        }

        let Some(line) = lines.next() else {
            // remember comments at the end of a block
            break full;
        };

        let mut push = |line| {
            full.push_str(line);
            trimmed.push_str(line.trim_start_matches(' ').trim_end_matches('\n'))
        };

        push(line);

        let indent = indentation(line);
        let mut multiline = false;

        // If the item is split between multiple lines...
        while let Some(more_indented) =
            lines.next_if(|&line: &&_| indent < indentation(line) || line == "\n")
        {
            multiline = true;
            push(more_indented);
        }

        if multiline
            && let Some(indented) =
                // Only append next indented line if it looks like a closing bracket.
                // Otherwise we incorrectly merge code like this (can be seen in
                // compiler/rustc_session/src/options.rs):
                //
                // force_unwind_tables: Option<bool> = (None, parse_opt_bool, [TRACKED],
                //     "force use of unwind tables"),
                // incremental: Option<String> = (None, parse_opt_string, [UNTRACKED],
                //     "enable incremental compilation"),
                lines.next_if(|l| {
                    indentation(l) == indent
                        && l.trim_start_matches(' ').starts_with(is_close_bracket)
                })
        {
            push(indented);
        }

        items.push(Item { full, trimmed });
    };

    items.sort_by(|a, b| version_sort(&a.trimmed, &b.trimmed));
    items.into_iter().map(|l| l.full).chain([end_comments]).collect()
}

fn check_lines<'a>(path: &Path, content: &'a str, tidy_ctx: &TidyCtx, check: &mut RunningCheck) {
    let mut offset = 0;

    loop {
        let rest = &content[offset..];
        let start = rest.find(START_MARKER);
        let end = rest.find(END_MARKER);

        match (start, end) {
            // error handling

            // end before start
            (Some(start), Some(end)) if end < start => {
                check.error(format!(
                    "{path}:{line_number} found `{END_MARKER}` expecting `{START_MARKER}`",
                    path = path.display(),
                    line_number = content[..offset + end].lines().count(),
                ));
                break;
            }

            // end without a start
            (None, Some(end)) => {
                check.error(format!(
                    "{path}:{line_number} found `{END_MARKER}` expecting `{START_MARKER}`",
                    path = path.display(),
                    line_number = content[..offset + end].lines().count(),
                ));
                break;
            }

            // start without an end
            (Some(start), None) => {
                check.error(format!(
                    "{path}:{line_number} `{START_MARKER}` without a matching `{END_MARKER}`",
                    path = path.display(),
                    line_number = content[..offset + start].lines().count(),
                ));
                break;
            }

            // a second start in between start/end pair
            (Some(start), Some(end))
                if rest[start + START_MARKER.len()..end].contains(START_MARKER) =>
            {
                check.error(format!(
                    "{path}:{line_number} found `{START_MARKER}` expecting `{END_MARKER}`",
                    path = path.display(),
                    line_number = content[..offset
                        + sub_find(rest, start + START_MARKER.len()..end, START_MARKER)
                            .unwrap()
                            .start]
                        .lines()
                        .count()
                ));
                break;
            }

            // happy happy path :3
            (Some(start), Some(end)) => {
                assert!(start <= end);

                // "...␤// tidy-alphabetical-start␤...␤// tidy-alphabetical-end␤..."
                //                  start_nl_end --^  ^-- end_nl_start          ^-- end_nl_end

                // Position after the newline after start marker
                let start_nl_end = sub_find(rest, start + START_MARKER.len().., "\n").unwrap().end;

                // Position before the new line before the end marker
                let end_nl_start = rest[..end].rfind('\n').unwrap();

                // Position after the newline after end marker
                let end_nl_end = sub_find(rest, end + END_MARKER.len().., "\n")
                    .map(|r| r.end)
                    .unwrap_or(content.len() - offset);

                let section = &rest[start_nl_end..=end_nl_start];
                let sorted = sort_section(section);

                // oh nyooo :(
                if sorted != section {
                    if !tidy_ctx.is_bless_enabled() {
                        let base_line_number = content[..offset + start_nl_end].lines().count();
                        let line_offset = sorted
                            .lines()
                            .zip(section.lines())
                            .enumerate()
                            .find(|(_, (a, b))| a != b)
                            .unwrap()
                            .0;
                        let line_number = base_line_number + line_offset;

                        check.error(format!(
                            "{path}:{line_number}: line not in alphabetical order (tip: use --bless to sort this list)",
                            path = path.display(),
                        ));
                    } else {
                        // Use atomic rename as to not corrupt the file upon crashes/ctrl+c
                        let mut tempfile =
                            tempfile::Builder::new().tempfile_in(path.parent().unwrap()).unwrap();

                        fs::copy(path, tempfile.path()).unwrap();

                        tempfile
                            .as_file_mut()
                            .seek(std::io::SeekFrom::Start((offset + start_nl_end) as u64))
                            .unwrap();
                        tempfile.as_file_mut().write_all(sorted.as_bytes()).unwrap();

                        tempfile.persist(path).unwrap();
                    }
                }

                // Start the next search after the end section
                offset += end_nl_end;
            }

            // No more alphabetical lists, yay :3
            (None, None) => break,
        }
    }
}

pub fn check(path: &Path, tidy_ctx: TidyCtx) {
    let mut check = tidy_ctx.start_check(CheckId::new("alphabetical").path(path));

    let skip = |path: &_, _is_dir| {
        filter_dirs(path)
            || path.ends_with("tidy/src/alphabetical.rs")
            || path.ends_with("tidy/src/alphabetical/tests.rs")
    };

    walk(path, skip, &mut |entry, content| {
        check_lines(entry.path(), content, &tidy_ctx, &mut check)
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

/// Finds `pat` in `s[range]` and returns a range such that `s[ret] == pat`.
fn sub_find(s: &str, range: impl RangeBounds<usize>, pat: &str) -> Option<Range<usize>> {
    s[(range.start_bound().cloned(), range.end_bound().cloned())]
        .find(pat)
        .map(|pos| {
            pos + match range.start_bound().cloned() {
                std::ops::Bound::Included(x) => x,
                std::ops::Bound::Excluded(x) => x + 1,
                std::ops::Bound::Unbounded => 0,
            }
        })
        .map(|pos| pos..pos + pat.len())
}
