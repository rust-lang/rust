use std::fmt;
use std::fs::File;
use std::io::BufReader;
use std::io::prelude::*;
use std::path::Path;
use std::str::FromStr;
use std::sync::OnceLock;

use regex::Regex;
use tracing::*;

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum ErrorKind {
    Help,
    Error,
    Note,
    Suggestion,
    Warning,
}

impl FromStr for ErrorKind {
    type Err = ();
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.to_uppercase();
        let part0: &str = s.split(':').next().unwrap();
        match part0 {
            "HELP" => Ok(ErrorKind::Help),
            "ERROR" => Ok(ErrorKind::Error),
            "NOTE" => Ok(ErrorKind::Note),
            "SUGGESTION" => Ok(ErrorKind::Suggestion),
            "WARN" | "WARNING" => Ok(ErrorKind::Warning),
            _ => Err(()),
        }
    }
}

impl fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            ErrorKind::Help => write!(f, "help message"),
            ErrorKind::Error => write!(f, "error"),
            ErrorKind::Note => write!(f, "note"),
            ErrorKind::Suggestion => write!(f, "suggestion"),
            ErrorKind::Warning => write!(f, "warning"),
        }
    }
}

#[derive(Debug)]
pub struct Error {
    pub line_num: Option<usize>,
    /// What kind of message we expect (e.g., warning, error, suggestion).
    /// `None` if not specified or unknown message kind.
    pub kind: Option<ErrorKind>,
    pub msg: String,
}

impl Error {
    pub fn render_for_expected(&self) -> String {
        use colored::Colorize;
        format!(
            "{: <10}line {: >3}: {}",
            self.kind.map(|kind| kind.to_string()).unwrap_or_default().to_uppercase(),
            self.line_num_str(),
            self.msg.cyan(),
        )
    }

    pub fn line_num_str(&self) -> String {
        self.line_num.map_or("?".to_string(), |line_num| line_num.to_string())
    }
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
pub fn load_errors(testfile: &Path, revision: Option<&str>) -> Vec<Error> {
    let rdr = BufReader::new(File::open(testfile).unwrap());

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
    let whole_match = captures.get(0).unwrap();
    let (_, mut msg) = line.split_at(whole_match.end());

    let first_word = msg.split_whitespace().next().expect("Encountered unexpected empty comment");

    // If we find `//~ ERROR foo` or something like that, skip the first word.
    let kind = first_word.parse::<ErrorKind>().ok();
    if kind.is_some() {
        msg = &msg.trim_start().split_at(first_word.len()).1;
    }

    let msg = msg.trim().to_owned();

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

    debug!(
        "line={:?} tag={:?} follow_prev={:?} kind={:?} msg={:?}",
        line_num,
        whole_match.as_str(),
        follow_prev,
        kind,
        msg
    );
    Some((follow_prev, Error { line_num, kind, msg }))
}

#[cfg(test)]
mod tests;
