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

    let mut directive = trimmed_comment.strip_prefix("@")?.trim_start();

    if let Some(revision) = leading_revision {
        let context = &line[..line.substr_range(directive).unwrap().start];
        return error(ErrorKind::LeadingRevisionCondition { revision, context });
    }

    if trimmed_comment.len() != comment.len() {
        return error(ErrorKind::WhitespaceBeforeAtSign);
    }

    let mut revision = None;

    if let Some(after_open_bracket) = directive.strip_prefix('[') {
        let Some((line_revision, after_close_bracket)) = after_open_bracket.split_once(']') else {
            return error(ErrorKind::UnclosedRevisionCondition);
        };

        revision = Some(line_revision);
        directive = after_close_bracket.trim_start();

        if line_revision.contains(",") {
            return error(ErrorKind::CommaInRevisionCondition {
                revision: line_revision,
                directive,
            });
        }
    }

    let mut kind: fn(_) -> _ = DirectivePayload::Value;
    let (name, payload) = directive
        .split_once(|char| {
            match char {
                ' ' => kind = DirectivePayload::Remark,
                ':' => {}
                _ => return false,
            };
            true
        })
        .map(|(name, payload)| (name, Some(kind(payload))))
        .unwrap_or((directive, None));

    Some(Ok(DirectiveLine { file_path, line_number, revision, name, payload }))
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

    /// The revision condition.
    ///
    /// The directive only applies to the given revision of a test.
    pub(crate) revision: Option<&'a str>,

    /// Name of the directive.
    pub(crate) name: &'a str,

    /// The playload of the directive.
    ///
    /// If the payload was separated using a colon (`:`) it will get represented as a
    /// [`DirectivePayload::Value`]. If the payload was separated using a space, it will be
    /// represented as a [`DirectivePayload::Remark`]. If the directive doesn't contain either colon
    /// or space, the payload will be `None`.
    pub(crate) payload: Option<DirectivePayload<'a>>,
}

impl<'ln> DirectiveLine<'ln> {
    pub(crate) fn synthetic(name: &'static str, file_path: &'ln Utf8Path) -> Self {
        Self { file_path, line_number: LineNumber::ZERO, revision: None, name, payload: None }
    }

    pub(crate) fn applies_to_test_revision(&self, test_revision: Option<&str>) -> bool {
        self.revision.is_none() || self.revision == test_revision
    }

    /// Display the name & payload of the directive.
    pub(crate) fn display(&self) -> impl fmt::Display {
        fmt::from_fn(|f| {
            f.write_str(self.name)?;
            if let Some(payload) = &self.payload {
                match payload {
                    DirectivePayload::Value(value) => write!(f, ":{value}"),
                    DirectivePayload::Remark(remark) => write!(f, " {remark}"),
                }?;
            }
            Ok(())
        })
    }
}

#[derive(Clone, Copy)]
#[cfg_attr(test, derive(Debug))]
pub(crate) enum DirectivePayload<'a> {
    Value(&'a str),
    Remark(&'a str),
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

        let title = |message| error!("malformed directive: {message} at {file_path}:{line_number}");

        match kind {
            ErrorKind::LeadingRevisionCondition { revision, context } => {
                title("found revision condition before `@` instead of after");
                help!("rewrite the start of the line to `//@[{revision}] ` (from `{context}`)");
            }
            ErrorKind::WhitespaceBeforeAtSign => {
                title("found whitespace between `//` and `@`");
                help!("if you meant to write a directive remove the whitespace to form `//@`");
                help!(
                    "if you meant to literally write `@` in a code comment add `\\` before `@` or \
                    rewrap the contents of the comment so `@` doesn't come first"
                );
            }
            ErrorKind::UnclosedRevisionCondition => {
                title("revision condition is missing a closing bracket `]`");
            }
            ErrorKind::CommaInRevisionCondition { revision, directive } => {
                title(
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
