// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
use self::WhichLine::*;

use std::fmt;
use std::fs::File;
use std::io::BufReader;
use std::io::prelude::*;
use std::path::Path;
use std::str::FromStr;

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
            "WARN" => Ok(ErrorKind::Warning),
            "WARNING" => Ok(ErrorKind::Warning),
            _ => Err(()),
        }
    }
}

impl fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            ErrorKind::Help => write!(f, "help"),
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
    /// What kind of message we expect (e.g. warning, error, suggestion).
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

    let tag = match cfg {
        Some(rev) => format!("//[{}]~", rev),
        None => format!("//~"),
    };

    rdr.lines()
        .enumerate()
        .filter_map(|(line_num, line)| {
            parse_expected(last_nonfollow_error, line_num + 1, &line.unwrap(), &tag)
                .map(|(which, error)| {
                    match which {
                        FollowPrevious(_) => {}
                        _ => last_nonfollow_error = Some(error.line_num),
                    }
                    error
                })
        })
        .collect()
}

fn parse_expected(last_nonfollow_error: Option<usize>,
                  line_num: usize,
                  line: &str,
                  tag: &str)
                  -> Option<(WhichLine, Error)> {
    let start = match line.find(tag) {
        Some(i) => i,
        None => return None,
    };
    let (follow, adjusts) = if line[start + tag.len()..].chars().next().unwrap() == '|' {
        (true, 0)
    } else {
        (false, line[start + tag.len()..].chars().take_while(|c| *c == '^').count())
    };
    let kind_start = start + tag.len() + adjusts + (follow as usize);
    let (kind, msg);
    match line[kind_start..]
        .split_whitespace()
        .next()
        .expect("Encountered unexpected empty comment")
        .parse::<ErrorKind>() {
        Ok(k) => {
            // If we find `//~ ERROR foo` or something like that:
            kind = Some(k);
            let letters = line[kind_start..].chars();
            msg = letters.skip_while(|c| c.is_whitespace())
                .skip_while(|c| !c.is_whitespace())
                .collect::<String>();
        }
        Err(_) => {
            // Otherwise we found `//~ foo`:
            kind = None;
            let letters = line[kind_start..].chars();
            msg = letters.skip_while(|c| c.is_whitespace())
                .collect::<String>();
        }
    }
    let msg = msg.trim().to_owned();

    let (which, line_num) = if follow {
        assert!(adjusts == 0, "use either //~| or //~^, not both.");
        let line_num = last_nonfollow_error.expect("encountered //~| without \
                                                    preceding //~^ line.");
        (FollowPrevious(line_num), line_num)
    } else {
        let which = if adjusts > 0 {
            AdjustBackward(adjusts)
        } else {
            ThisLine
        };
        let line_num = line_num - adjusts;
        (which, line_num)
    };

    debug!("line={} tag={:?} which={:?} kind={:?} msg={:?}",
           line_num,
           tag,
           which,
           kind,
           msg);
    Some((which,
          Error {
        line_num: line_num,
        kind: kind,
        msg: msg,
    }))
}
