use std::fmt;

use camino::Utf8Path;

use crate::directives::LineNumber;
use crate::{error, fatal, help};

/// If the given line begins with the appropriate comment prefix for a directive,
/// returns a struct containing various parts of the directive.
pub(crate) fn line_directive<'a>(
    file_path: &'a Utf8Path,
    line_number: LineNumber,
    line: &'a str,
) -> Option<DirectiveLine<'a>> {
    let comment = line.trim_start().strip_prefix("//")?;
    let trimmed_comment = comment.trim_start();

    let (leading_revision, trimmed_comment) = trimmed_comment
        .strip_prefix('[')
        .and_then(|after_open_bracket| after_open_bracket.split_once(']'))
        .map(|(revision, after_close_bracket)| (Some(revision), after_close_bracket.trim_start()))
        .unwrap_or((None, trimmed_comment));

    let mut raw_directive = trimmed_comment.strip_prefix("@")?.trim_start();

    if let Some(leading_revision) = leading_revision {
        error!(
            "malformed directive: found revision condition before `@` instead of after at {file_path}:{line_number}"
        );
        let from = &line[..line.substr_range(raw_directive).unwrap().start];
        help!("rewrite the start of the line to `//@[{leading_revision}] ` (from `{from}`)");
        panic!("fatal error"); // FIXME: Poison the context and keep going.
    }

    if trimmed_comment.len() != comment.len() {
        error!(
            "malformed directive: \
            found whitespace between `//` and `@` at {file_path}:{line_number}"
        );
        help!("if you meant to write a directive remove the whitespace to form `//@`");
        help!(
            "if you meant to literally write `@` in a code comment \
            add `\\` before `@` or rewrap the contents of the comment so `@` doesn't come first"
        );
        panic!("fatal error"); // FIXME: Poison the context and keep going.
    }

    let mut revision = None;

    if let Some(after_open_bracket) = raw_directive.strip_prefix('[') {
        // A comment like `//@[foo]` only applies to revision `foo`.
        let Some((line_revision, after_close_bracket)) = after_open_bracket.split_once(']') else {
            fatal!("malformed condition directive: expected `//@[foo]`, found `{line}`")
        };

        revision = Some(line_revision);
        raw_directive = after_close_bracket.trim_start();

        if line_revision.contains(",") {
            let suggestion: Vec<_> = line_revision
                .split(",")
                .map(|revision| format!("//@[{revision}]: {raw_directive}"))
                .collect();
            panic!(
                "malformed condition directive: multiple revisions aren't supported yet in `{}`, split them like\n{}",
                line,
                suggestion.join("\n"),
            );
        }
    }

    // The directive name ends at the first occurrence of colon, space, or end-of-string.
    let name = raw_directive.split([':', ' ']).next().expect("split is never empty");

    Some(DirectiveLine { file_path, line_number, revision, raw_directive, name })
}

/// The (partly) broken-down contents of a line containing a test directive,
/// which `iter_directives` passes to its callback function.
///
/// For example:
///
/// ```text
/// //@ compile-flags: -O
///     ^^^^^^^^^^^^^^^^^ raw_directive
///     ^^^^^^^^^^^^^     name
///
/// //@ [foo] compile-flags: -O
///      ^^^                    revision
///           ^^^^^^^^^^^^^^^^^ raw_directive
///           ^^^^^^^^^^^^^     name
/// ```
pub(crate) struct DirectiveLine<'a> {
    /// Path of the file containing this line.
    ///
    /// Mostly used for diagnostics, but some directives (e.g. `//@ pp-exact`)
    /// also use it to compute a value based on the filename.
    pub(crate) file_path: &'a Utf8Path,
    pub(crate) line_number: LineNumber,

    /// Some test directives start with a revision name in square brackets
    /// (e.g. `[foo]`), and only apply to that revision of the test.
    /// If present, this field contains the revision name (e.g. `foo`).
    pub(crate) revision: Option<&'a str>,

    /// The main part of the directive, after removing the comment prefix
    /// and the optional revision specifier.
    ///
    /// This is "raw" because the directive's name and colon-separated value
    /// (if present) have not yet been extracted or checked.
    raw_directive: &'a str,

    /// Name of the directive.
    ///
    /// Invariant: `self.raw_directive.starts_with(self.name)`
    pub(crate) name: &'a str,
}

impl<'ln> DirectiveLine<'ln> {
    pub(crate) fn applies_to_test_revision(&self, test_revision: Option<&str>) -> bool {
        self.revision.is_none() || self.revision == test_revision
    }

    /// Helper method used by `value_after_colon` and `remark_after_space`.
    /// Don't call this directly.
    fn rest_after_separator(&self, separator: u8) -> Option<&'ln str> {
        let n = self.name.len();
        if self.raw_directive.as_bytes().get(n) != Some(&separator) {
            return None;
        }

        Some(&self.raw_directive[n + 1..])
    }

    /// If this directive uses `name: value` syntax, returns the part after
    /// the colon character.
    pub(crate) fn value_after_colon(&self) -> Option<&'ln str> {
        self.rest_after_separator(b':')
    }

    /// If this directive uses `name remark` syntax, returns the part after
    /// the separating space.
    pub(crate) fn remark_after_space(&self) -> Option<&'ln str> {
        self.rest_after_separator(b' ')
    }

    /// Allows callers to print `raw_directive` if necessary,
    /// without accessing the field directly.
    pub(crate) fn display(&self) -> impl fmt::Display {
        self.raw_directive
    }
}
