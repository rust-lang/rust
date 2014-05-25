// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::io::{BufferedReader, File};
use regex::Regex;

pub struct ExpectedError {
    pub line: uint,
    pub kind: String,
    pub msg: String,
}

pub static EXPECTED_PATTERN : &'static str = r"//~(?P<adjusts>\^*)\s*(?P<kind>\S*)\s*(?P<msg>.*)";

// Load any test directives embedded in the file
pub fn load_errors(re: &Regex, testfile: &Path) -> Vec<ExpectedError> {
    let mut rdr = BufferedReader::new(File::open(testfile).unwrap());

    rdr.lines().enumerate().filter_map(|(line_no, ln)| {
        parse_expected(line_no + 1, ln.unwrap().as_slice(), re)
    }).collect()
}

fn parse_expected(line_num: uint, line: &str, re: &Regex) -> Option<ExpectedError> {
    re.captures(line).and_then(|caps| {
        let adjusts = caps.name("adjusts").len();
        let kind = caps.name("kind").to_ascii().to_lower().into_str().to_string();
        let msg = caps.name("msg").trim().to_string();

        debug!("line={} kind={} msg={}", line_num, kind, msg);
        Some(ExpectedError {
            line: line_num - adjusts,
            kind: kind,
            msg: msg,
        })
    })
}
