use self::WhichLine::*;

use std::fmt;
use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;
use std::path::Path;
use std::str::FromStr;

use once_cell::sync::Lazy;
use regex::Regex;
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

/// Looks for either "//~| KIND MESSAGE" or "//~^^... KIND MESSAGE"
/// The former is a "follow" that inherits its target from the preceding line;
/// the latter is an "adjusts" that goes that many lines up.
///
/// Goal is to enable tests both like: //~^^^ ERROR go up three
/// and also //~^ ERROR message one for the preceding line, and
///          //~| ERROR message two for that same line.
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
            parse_expected(last_nonfollow_error, line_num + 1, &line.unwrap(), cfg).map(
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

fn parse_expected(
    last_nonfollow_error: Option<usize>,
    line_num: usize,
    line: &str,
    cfg: Option<&str>,
) -> Option<(WhichLine, Error)> {
    // Matches comments like:
    //     //~
    //     //~|
    //     //~^
    //     //~^^^^^
    //     //[cfg1]~
    //     //[cfg1,cfg2]~^^
    static RE: Lazy<Regex> =
        Lazy::new(|| Regex::new(r"//(?:\[(?P<cfgs>[\w,]+)])?~(?P<adjust>\||\^*)").unwrap());

    let captures = RE.captures(line)?;

    match (cfg, captures.name("cfgs")) {
        // Only error messages that contain our `cfg` between the square brackets apply to us.
        (Some(cfg), Some(filter)) if !filter.as_str().split(',').any(|s| s == cfg) => return None,
        (Some(_), Some(_)) => {}

        (None, Some(_)) => panic!("Only tests with revisions should use `//[X]~`"),

        // If an error has no list of revisions, it applies to all revisions.
        (Some(_), None) | (None, None) => {}
    }

    let (follow, adjusts) = match &captures["adjust"] {
        "|" => (true, 0),
        circumflexes => (false, circumflexes.len()),
    };

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

    let (which, line_num) = if follow {
        assert_eq!(adjusts, 0, "use either //~| or //~^, not both.");
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

    debug!(
        "line={} tag={:?} which={:?} kind={:?} msg={:?}",
        line_num,
        whole_match.as_str(),
        which,
        kind,
        msg
    );
    Some((which, Error { line_num, kind, msg }))
}
