use std::fmt;

use camino::Utf8Path;

use crate::directives::LineNumber;
use crate::{error, help};

/// If the given line begins with the appropriate comment prefix for a directive,
/// returns a struct containing various parts of the directive.
pub(crate) fn line_directive<'a>(
    file_path: &'a Utf8Path,
    line_number: LineNumber,
    line: &'a str,
) -> Option<Result<DirectiveLine<'a>, Error<'a>>> {
    let error = |kind| Some(Err(Error { kind, file_path, line_number }));

    let comment = line.trim_start().strip_prefix("//")?;
    let trimmed_comment = comment.trim_start();

    let (leading_revision, trimmed_comment) = trimmed_comment
        .strip_prefix('[')
        .and_then(|after_open_bracket| after_open_bracket.split_once(']'))
        .map(|(revision, after_close_bracket)| (Some(revision), after_close_bracket.trim_start()))
        .unwrap_or((None, trimmed_comment));

    let mut raw_directive = trimmed_comment.strip_prefix("@")?.trim_start();

    if let Some(revision) = leading_revision {
        let context = &line[..line.substr_range(raw_directive).unwrap().start];
        return error(ErrorKind::LeadingRevisionCondition { revision, context });
    }

    if trimmed_comment.len() != comment.len() {
        return error(ErrorKind::WhitespaceBeforeAtSign);
    }

    let mut revision = None;

    if let Some(after_open_bracket) = raw_directive.strip_prefix('[') {
        let Some((line_revision, after_close_bracket)) = after_open_bracket.split_once(']') else {
            return error(ErrorKind::UnclosedRevisionCondition);
        };

        revision = Some(line_revision);
        raw_directive = after_close_bracket.trim_start();

        if line_revision.contains(",") {
            return error(ErrorKind::CommaInRevisionCondition {
                revision: line_revision,
                directive: raw_directive,
            });
        }
    }

    // The directive name ends at the first occurrence of colon, space, or end-of-string.
    let name = raw_directive.split([':', ' ']).next().expect("split is never empty");

    Some(Ok(DirectiveLine { file_path, line_number, revision, raw_directive, name }))
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
#[cfg_attr(test, derive(Debug))]
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
    #[cfg(not(test))]
    raw_directive: &'a str,
    #[cfg(test)]
    pub(crate) raw_directive: &'a str,

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

#[cfg_attr(test, derive(Debug))]
pub(crate) struct Error<'a> {
    pub(crate) kind: ErrorKind<'a>,
    pub(crate) file_path: &'a Utf8Path,
    pub(crate) line_number: LineNumber,
}

#[cfg_attr(test, derive(Debug))]
pub(crate) enum ErrorKind<'a> {
    LeadingRevisionCondition { revision: &'a str, context: &'a str },
    WhitespaceBeforeAtSign,
    UnclosedRevisionCondition,
    CommaInRevisionCondition { revision: &'a str, directive: &'a str },
}

impl Error<'_> {
    // FIXME: Just report the error, poison the context and keep going.
    pub(crate) fn raise_fatal(self) -> ! {
        let Self { kind, file_path, line_number } = self;

        let error = |message| error!("malformed directive: {message} at {file_path}:{line_number}");

        match kind {
            ErrorKind::LeadingRevisionCondition { revision, context } => {
                error("found revision condition before `@` instead of after");
                help!("rewrite the start of the line to `//@[{revision}] ` (from `{context}`)");
            }
            ErrorKind::WhitespaceBeforeAtSign => {
                error("found whitespace between `//` and `@`");
                help!("if you meant to write a directive remove the whitespace to form `//@`");
                help!(
                    "if you meant to literally write `@` in a code comment add `\\` before `@` or \
                    rewrap the contents of the comment so `@` doesn't come first"
                );
            }
            ErrorKind::UnclosedRevisionCondition => {
                error("revision condition is missing a closing bracket `]`");
            }
            ErrorKind::CommaInRevisionCondition { revision, directive } => {
                error(
                    "found commas in revision condition but multiple revisions aren't supported yet",
                );

                let suggestion = revision
                    .split(",")
                    .map(|revision| format!("//@[{revision}]: {directive}"))
                    .collect::<Vec<_>>()
                    .join("\n");

                help!("consider splitting the directive like\n{suggestion}");
            }
        }

        panic!("fatal error")
    }
}
