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

use std::ascii::AsciiExt;
use std::io::{BufferedReader, File};
use regex::Regex;

pub struct ExpectedError {
    pub line: uint,
    pub kind: String,
    pub msg: String,
}

/// Looks for either "//~| KIND MESSAGE" or "//~^^... KIND MESSAGE"
/// The former is a "follow" that inherits its target from the preceding line;
/// the latter is an "adjusts" that goes that many lines up.
///
/// Goal is to enable tests both like: //~^^^ ERROR go up three
/// and also //~^ ERROR message one for the preceding line, and
///          //~| ERROR message two for that same line.

pub static EXPECTED_PATTERN : &'static str =
    r"//~(?P<follow>\|)?(?P<adjusts>\^*)\s*(?P<kind>\S*)\s*(?P<msg>.*)";

#[derive(PartialEq, Show)]
enum WhichLine { ThisLine, FollowPrevious(uint), AdjustBackward(uint) }

// Load any test directives embedded in the file
pub fn load_errors(re: &Regex, testfile: &Path) -> Vec<ExpectedError> {
    let mut rdr = BufferedReader::new(File::open(testfile).unwrap());

    // `last_nonfollow_error` tracks the most recently seen
    // line with an error template that did not use the
    // follow-syntax, "//~| ...".
    //
    // (pnkfelix could not find an easy way to compose Iterator::scan
    // and Iterator::filter_map to pass along this information into
    // `parse_expected`. So instead I am storing that state here and
    // updating it in the map callback below.)
    let mut last_nonfollow_error = None;

    rdr.lines().enumerate().filter_map(|(line_no, ln)| {
        parse_expected(last_nonfollow_error,
                       line_no + 1,
                       ln.unwrap().as_slice(), re)
            .map(|(which, error)| {
                match which {
                    FollowPrevious(_) => {}
                    _ => last_nonfollow_error = Some(error.line),
                }
                error
            })
    }).collect()
}

fn parse_expected(last_nonfollow_error: Option<uint>,
                  line_num: uint,
                  line: &str,
                  re: &Regex) -> Option<(WhichLine, ExpectedError)> {
    re.captures(line).and_then(|caps| {
        let adjusts = caps.name("adjusts").unwrap_or("").len();
        let kind = caps.name("kind").unwrap_or("").to_ascii_lowercase();
        let msg = caps.name("msg").unwrap_or("").trim().to_string();
        let follow = caps.name("follow").unwrap_or("").len() > 0;

        let (which, line) = if follow {
            assert!(adjusts == 0, "use either //~| or //~^, not both.");
            let line = last_nonfollow_error.unwrap_or_else(|| {
                panic!("encountered //~| without preceding //~^ line.")
            });
            (FollowPrevious(line), line)
        } else {
            let which =
                if adjusts > 0 { AdjustBackward(adjusts) } else { ThisLine };
            let line = line_num - adjusts;
            (which, line)
        };

        debug!("line={} which={:?} kind={:?} msg={:?}", line_num, which, kind, msg);
        Some((which, ExpectedError { line: line,
                                     kind: kind,
                                     msg: msg, }))
    })
}
