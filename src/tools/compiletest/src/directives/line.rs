use std::fmt;

use camino::Utf8Path;

use crate::directives::{LineNumber, Location};
use crate::{error, help};

/// If the given line begins with the appropriate comment prefix for a directive,
/// returns a struct containing various parts of the directive.
pub(crate) fn line_directive<'a>(
    location: Location<'a>,
    line: &'a str,
) -> Option<Result<DirectiveLine<'a>, Error<'a>>> {
    let error = |kind| Some(Err(Error { kind, location }));

    let comment = line.trim().strip_prefix("//")?;
    let trimmed_comment = comment.trim_start();

    // FIXME: Account for the fact that we now require revision names to be valid identifiers.
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
        let Some((rev, after_close_bracket)) = after_open_bracket.split_once(']') else {
            return error(ErrorKind::UnclosedRevisionCondition);
        };

        directive = after_close_bracket.trim_start();

        let rev = rev.trim();
        if rev.contains(",") {
            return error(ErrorKind::CommaInRevisionCondition { revision: rev, directive });
        }
        if !super::is_valid_revision_name(rev) {
            return error(ErrorKind::InvalidRevisionName { revision: rev });
        }

        revision = Some(rev);
    }

    let (name, payload) = directive
        .find(|char: char| char == SEPARATOR || char.is_whitespace())
        .map(|index| (&directive[..index], Some(directive[index..].trim_start())))
        .unwrap_or((directive, None));

    if name.is_empty() {
        return error(ErrorKind::MissingDirectiveName);
    }

    let payload = if let Some(payload) = payload {
        if let Some(value) = payload.strip_prefix(SEPARATOR) {
            let value = value.trim_start();
            if value.is_empty() {
                return error(ErrorKind::MissingValue);
            }
            Some(DirectivePayload::Value(value))
        } else {
            Some(DirectivePayload::Remark(payload))
        }
    } else {
        None
    };

    Some(Ok(DirectiveLine { location, revision, name, payload }))
}

const SEPARATOR: char = ':';

/// The broken-down contents of a test directive.
///
/// For example:
///
/// ```text
/// //@ compile-flags: -O
///     ^^^^^^^^^^^^^     name
///                    ^^ payload (Value)
///
/// //@ [foo] compile-flags: -O
///      ^^^                    revision
///           ^^^^^^^^^^^^^     name
///                          ^^ payload (Value)
///
/// //@ only-unix because SIGPIPE is a unix thing
///     ^^^^^^^^^                                 name
///               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ payload (Remark)
/// ```
#[cfg_attr(test, derive(Debug))]
pub(crate) struct DirectiveLine<'a> {
    /// The location of this directive line.
    ///
    /// Mostly used for diagnostics, but some directives (e.g. `//@ pp-exact`)
    /// also use it to compute a value based on the filename.
    pub(crate) location: Location<'a>,

    /// The revision condition.
    ///
    /// The directive only applies to the given revision of a test.
    pub(crate) revision: Option<&'a str>,

    /// Name of the directive.
    pub(crate) name: &'a str,

    /// The non-empty whitespace-trimmed payload of the directive.
    ///
    /// If the payload was separated using a colon (`:`), it's a [`DirectivePayload::Value`].
    /// If the payload was separated using whitespace, it's a [`DirectivePayload::Remark`].
    pub(crate) payload: Option<DirectivePayload<'a>>,
}

impl<'ln> DirectiveLine<'ln> {
    // FIXME: Get rid of synthetic directives.
    //        See also <https://github.com/rust-lang/rust/pull/165651#discussion_r4541102589>.
    pub(crate) fn synthetic(name: &'static str, file_path: &'ln Utf8Path) -> Self {
        let location = Location { file_path, line_number: LineNumber::ZERO };
        Self { location, revision: None, name, payload: None }
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
                    DirectivePayload::Value(value) => write!(f, ": {value}"),
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
    pub(crate) location: Location<'a>,
}

#[cfg_attr(test, derive(Debug))]
pub(crate) enum ErrorKind<'a> {
    LeadingRevisionCondition { revision: &'a str, context: &'a str },
    WhitespaceBeforeAtSign,
    UnclosedRevisionCondition,
    CommaInRevisionCondition { revision: &'a str, directive: &'a str },
    MissingValue,
    MissingDirectiveName,
    InvalidRevisionName { revision: &'a str },
}

impl Error<'_> {
    // FIXME: Just report the error, poison the context and keep going.
    pub(crate) fn raise_fatal(self) -> ! {
        let title = |message| error!("malformed directive: {message} at {}", self.location);

        match self.kind {
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
            ErrorKind::MissingValue => {
                title("a value is missing following the `:`");

                help!("consider removing the colon if you didn't mean to specify any value");
                help!("consider providing a value after the `:` if you meant to provide one");
            }
            ErrorKind::MissingDirectiveName => title("a directive name is missing"),
            ErrorKind::InvalidRevisionName { revision } => {
                title(&format!("`{revision}` is not a valid revision name"));

                help!("the name has to be {}", super::DESCR_OF_VALID_REVISION_NAMES);
            }
        }

        panic!("fatal error")
    }
}
