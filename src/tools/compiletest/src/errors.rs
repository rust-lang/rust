use self::WhichLine::*;

use std::fmt;
use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;
use std::path::Path;
use std::str::FromStr;

use once_cell::sync::Lazy;
use regex::{Captures, Regex};
use tracing::*;

#[derive(Clone, Debug, PartialEq)]
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
    pub line_num: usize,
    /// What kind of message we expect (e.g., warning, error, suggestion).
    /// `None` if not specified or unknown message kind.
    pub kind: Option<ErrorKind>,
    pub msg: String,
}

#[derive(PartialEq, Debug)]
enum WhichLine {
    ThisLine,
    FollowPrevious(usize),
    AdjustBackward(usize),
}

/// Looks for either `//~| KIND MESSAGE` or `//~^^... KIND MESSAGE`
/// The former is a "follow" that inherits its target from the preceding line;
/// the latter is an "adjusts" that goes that many lines up.
///
/// Goal is to enable tests both like: `//~^^^ ERROR` go up three
/// and also `//~^` ERROR message one for the preceding line, and
///          `//~|` ERROR message two for that same line.
///
/// If cfg is not None (i.e., in an incremental test), then we look
/// for `//[X]~` instead, where `X` is the current `cfg`.
pub fn load_errors(testfile: &Path, cfg: Option<&str>) -> Vec<Error> {
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
        .filter_map(|(line_num, line)| {
            try_parse_error_comment(last_nonfollow_error, line_num + 1, &line.unwrap(), cfg).map(
                |(which, error)| {
                    match which {
                        FollowPrevious(_) => {}
                        _ => last_nonfollow_error = Some(error.line_num),
                    }

                    error
                },
            )
        })
        .collect()
}

/// Parses an error pattern from a line, if a pattern exists on that line.
fn try_parse_error_comment(
    last_nonfollow_error: Option<usize>,
    line_num: usize,
    line: &str,
    cfg: Option<&str>,
) -> Option<(WhichLine, Error)> {
    let mut line = line.trim_start();

    // compiletest style revisions are `[revs]~`
    static COMPILETEST_REVISION: Lazy<Regex> =
        Lazy::new(|| Regex::new(r"//\[(?P<revs>[\w,]+)\]~").unwrap());

    // ui_test style revisions are `~[revs]`
    static UI_TEST_REVISION: Lazy<Regex> =
        Lazy::new(|| Regex::new(r"//~\[(?P<revs>[\w,]+)\]").unwrap());

    let check_valid_rev = |captures: &Captures<'_>| {
        let revs = captures.name("revs").unwrap_or_else(|| {
            panic!("expected comment {} parsed as compiletest to have a revs group", line)
        });
        match cfg {
            // If the comment has revisions, only emit an expected error if one of the specified
            // revisions is the current revision.
            Some(current_rev) => {
                revs.as_str().split(',').position(|rev| rev == current_rev).is_some()
            }
            None => {
                panic!("Only tests with revisions should use revisioned error patterns //~[rev]")
            }
        }
    };

    // Check for the different types of revisions.
    // If neither of the revision styles match, it's a normal error pattern which must start with a //~
    // Note that error pattern comments may start anywhere within a line, such as on the same line as code.
    if let Some(captures) = COMPILETEST_REVISION.captures(line) {
        if !check_valid_rev(&captures) {
            // Comment doesn't have a revision for the current revision.
            return None;
        }
        // Remove the matched revisions and trailing ~ from the line.
        line = &line[captures.get(0).unwrap().end()..];
    } else if let Some(captures) = UI_TEST_REVISION.captures(line) {
        if !check_valid_rev(&captures) {
            // Comment doesn't have a revision for the current revision.
            return None;
        }
        // Remove the matched ~ and revisions from the line.
        line = &line[captures.get(0).unwrap().end()..];
    } else {
        // Errors without revisions start with a //~ so find where that starts
        line = line.find("//~").map(|idx| &line[idx + 3..])?;
    }

    // At this point, if the comment has revisions, they've been verified to be correct for the
    // current checking revision. Those revisions have been stripped if applicable, and the leading
    // ~ for non-revisioned comments has been removed.

    // Parse adjustments:
    //  - | = "same line as previous error"
    //  - ^ = "applies to the previous line" (may be repeated indefinitely)
    // Only one type of adjustment may exist per error pattern.

    let (follow, adjusts) = if line.starts_with('|') {
        line = &line[1..];
        (true, 0)
    } else {
        let adjust_count = line.chars().take_while(|&c| c == '^').count();
        line = &line[adjust_count..];
        (false, adjust_count)
    };

    line = line.trim_start();
    let first_word = line.split_whitespace().next().expect("Encountered unexpected empty comment");

    // If we find `//~ ERROR foo` or something like that, skip the first word.
    // The `FromStr` impl for ErrorKind accepts a trailing `:` too.
    let kind = first_word.parse::<ErrorKind>().ok();
    if kind.is_some() {
        line = &line.trim_start().split_at(first_word.len()).1;
    }

    let line = line.trim().to_owned();

    let (which, line_num) = if follow {
        let line_num = last_nonfollow_error.expect(
            "encountered //~| without \
             preceding //~^ line.",
        );
        (FollowPrevious(line_num), line_num)
    } else {
        let which = if adjusts > 0 { AdjustBackward(adjusts) } else { ThisLine };
        let line_num = line_num - adjusts;
        (which, line_num)
    };

    debug!("line={} which={:?} kind={:?} line={:?}", line_num, which, kind, line);
    Some((which, Error { line_num, kind, msg: line }))
}
