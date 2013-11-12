// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::io::buffered::BufferedReader;
use std::io::File;

pub struct ExpectedError { line: uint, kind: ~str, msg: ~str }

// Load any test directives embedded in the file
pub fn load_errors(testfile: &Path) -> ~[ExpectedError] {

    let mut error_patterns = ~[];
    let mut rdr = BufferedReader::new(File::open(testfile).unwrap());
    let mut line_num = 1u;
    loop {
        let ln = match rdr.read_line() {
            Some(ln) => ln, None => break,
        };
        error_patterns.push_all_move(parse_expected(line_num, ln));
        line_num += 1u;
    }
    return error_patterns;
}

fn parse_expected(line_num: uint, line: ~str) -> ~[ExpectedError] {
    let line = line.trim();
    let error_tag = ~"//~";
    let mut idx;
    match line.find_str(error_tag) {
      None => return ~[],
      Some(nn) => { idx = (nn as uint) + error_tag.len(); }
    }

    // "//~^^^ kind msg" denotes a message expected
    // three lines above current line:
    let mut adjust_line = 0u;
    let len = line.len();
    while idx < len && line[idx] == ('^' as u8) {
        adjust_line += 1u;
        idx += 1u;
    }

    // Extract kind:
    while idx < len && line[idx] == (' ' as u8) { idx += 1u; }
    let start_kind = idx;
    while idx < len && line[idx] != (' ' as u8) { idx += 1u; }

    let kind = line.slice(start_kind, idx);
    let kind = kind.to_ascii().to_lower().into_str();

    // Extract msg:
    while idx < len && line[idx] == (' ' as u8) { idx += 1u; }
    let msg = line.slice(idx, len).to_owned();

    debug!("line={} kind={} msg={}", line_num - adjust_line, kind, msg);

    return ~[ExpectedError{line: line_num - adjust_line, kind: kind,
                           msg: msg}];
}
