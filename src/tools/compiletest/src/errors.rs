use std::fmt;
use std::fs::File;
use std::io::BufReader;
use std::io::prelude::*;
use std::sync::OnceLock;

use camino::Utf8Path;
use regex::Regex;
use tracing::*;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ErrorKind {
    Help,
    Error,
    Note,
    Suggestion,
    Warning,
    Raw,
    /// Used for better recovery and diagnostics in compiletest.
    Unknown,
}

impl ErrorKind {
    pub fn from_compiler_str(s: &str) -> ErrorKind {
        match s {
            "help" => ErrorKind::Help,
            "error" | "error: internal compiler error" => ErrorKind::Error,
            "note" | "failure-note" => ErrorKind::Note,
            "warning" => ErrorKind::Warning,
            _ => panic!("unexpected compiler diagnostic kind `{s}`"),
        }
    }

    /// Either the canonical uppercase string, or some additional versions for compatibility.
    /// FIXME: consider keeping only the canonical versions here.
    fn from_user_str(s: &str) -> Option<ErrorKind> {
        Some(match s {
            "HELP" | "help" => ErrorKind::Help,
            "ERROR" | "error" => ErrorKind::Error,
            "NOTE" | "note" => ErrorKind::Note,
            "SUGGESTION" => ErrorKind::Suggestion,
            "WARN" | "WARNING" | "warn" | "warning" => ErrorKind::Warning,
            "RAW" => ErrorKind::Raw,
            _ => return None,
        })
    }

    pub fn expect_from_user_str(s: &str) -> ErrorKind {
        ErrorKind::from_user_str(s).unwrap_or_else(|| {
            panic!(
                "unexpected diagnostic kind `{s}`, expected \
                 `ERROR`, `WARN`, `NOTE`, `HELP`, `SUGGESTION` or `RAW`"
            )
        })
    }
}

impl fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            ErrorKind::Help => write!(f, "HELP"),
            ErrorKind::Error => write!(f, "ERROR"),
            ErrorKind::Note => write!(f, "NOTE"),
            ErrorKind::Suggestion => write!(f, "SUGGESTION"),
            ErrorKind::Warning => write!(f, "WARN"),
            ErrorKind::Raw => write!(f, "RAW"),
            ErrorKind::Unknown => write!(f, "UNKNOWN"),
        }
    }
}

#[derive(Debug)]
pub struct Error {
    pub line_num: Option<usize>,
    pub column_num: Option<usize>,
    /// What kind of message we expect (e.g., warning, error, suggestion).
    pub kind: ErrorKind,
    pub msg: String,
    /// For some `Error`s, like secondary lines of multi-line diagnostics, line annotations
    /// are not mandatory, even if they would otherwise be mandatory for primary errors.
    /// Only makes sense for "actual" errors, not for "expected" errors.
    pub require_annotation: bool,
}

/// Looks for either "//~| KIND MESSAGE" or "//~^^... KIND MESSAGE"
/// The former is a "follow" that inherits its target from the preceding line;
/// the latter is an "adjusts" that goes that many lines up.
///
/// Goal is to enable tests both like: //~^^^ ERROR go up three
/// and also //~^ ERROR message one for the preceding line, and
///          //~| ERROR message two for that same line.
///
/// If revision is not None, then we look
/// for `//[X]~` instead, where `X` is the current revision.
pub fn load_errors(testfile: &Utf8Path, revision: Option<&str>) -> Vec<Error> {
    let rdr = BufReader::new(File::open(testfile.as_std_path()).unwrap());

    // `last_nonfollow_error` tracks the most recently seen
    // line with an error template that did not use the
    // follow-syntax, "//~| ...".
    //
    // (pnkfelix could not find an easy way to compose Iterator::scan
    // and Iterator::filter_map to pass along this information into
    // `parse_expected`. So instead I am storing that state here and
    // updating it in the map callback below.)
    let mut last_nonfollow_error = None;

    rdr.lines()
        .enumerate()
        // We want to ignore utf-8 failures in tests during collection of annotations.
        .filter(|(_, line)| line.is_ok())
        .filter_map(|(line_num, line)| {
            parse_expected(last_nonfollow_error, line_num + 1, &line.unwrap(), revision).map(
                |(follow_prev, error)| {
                    if !follow_prev {
                        last_nonfollow_error = error.line_num;
                    }
                    error
                },
            )
        })
        .collect()
}

fn parse_expected(
    last_nonfollow_error: Option<usize>,
    line_num: usize,
    line: &str,
    test_revision: Option<&str>,
) -> Option<(bool, Error)> {
    // Matches comments like:
    //     //~
    //     //~|
    //     //~^
    //     //~^^^^^
    //     //~v
    //     //~vvvvv
    //     //~?
    //     //[rev1]~
    //     //[rev1,rev2]~^^
    static RE: OnceLock<Regex> = OnceLock::new();

    let captures = RE
        .get_or_init(|| {
            Regex::new(r"//(?:\[(?P<revs>[\w\-,]+)])?~(?P<adjust>\?|\||[v\^]*)").unwrap()
        })
        .captures(line)?;

    match (test_revision, captures.name("revs")) {
        // Only error messages that contain our revision between the square brackets apply to us.
        (Some(test_revision), Some(revision_filters)) => {
            if !revision_filters.as_str().split(',').any(|r| r == test_revision) {
                return None;
            }
        }

        (None, Some(_)) => panic!("Only tests with revisions should use `//[X]~`"),

        // If an error has no list of revisions, it applies to all revisions.
        (Some(_), None) | (None, None) => {}
    }

    // Get the part of the comment after the sigil (e.g. `~^^` or ~|).
    let tag = captures.get(0).unwrap();
    let rest = line[tag.end()..].trim_start();
    let (kind_str, _) =
        rest.split_once(|c: char| c != '_' && !c.is_ascii_alphabetic()).unwrap_or((rest, ""));
    let (kind, untrimmed_msg) = match ErrorKind::from_user_str(kind_str) {
        Some(kind) => (kind, &rest[kind_str.len()..]),
        None => (ErrorKind::Unknown, rest),
    };
    let msg = untrimmed_msg.strip_prefix(':').unwrap_or(untrimmed_msg).trim().to_owned();

    let line_num_adjust = &captures["adjust"];
    let (follow_prev, line_num) = if line_num_adjust == "|" {
        (true, Some(last_nonfollow_error.expect("encountered //~| without preceding //~^ line")))
    } else if line_num_adjust == "?" {
        (false, None)
    } else if line_num_adjust.starts_with('v') {
        (false, Some(line_num + line_num_adjust.len()))
    } else {
        (false, Some(line_num - line_num_adjust.len()))
    };
    let column_num = Some(tag.start() + 1);

    debug!(
        "line={:?} tag={:?} follow_prev={:?} kind={:?} msg={:?}",
        line_num,
        tag.as_str(),
        follow_prev,
        kind,
        msg
    );
    Some((follow_prev, Error { line_num, column_num, kind, msg, require_annotation: true }))
}

#[cfg(test)]
mod tests;
