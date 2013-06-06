// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::prelude::*;

use core::io;
use core::str;

pub struct ExpectedError { line: uint, kind: ~str, msg: ~str }

// Load any test directives embedded in the file
pub fn load_errors(testfile: &Path) -> ~[ExpectedError] {
    let mut error_patterns = ~[];
    let rdr = io::file_reader(testfile).get();
    let mut line_num = 1u;
    while !rdr.eof() {
        let ln = rdr.read_line();
        error_patterns += parse_expected(line_num, ln);
        line_num += 1u;
    }
    return error_patterns;
}

fn parse_expected(line_num: uint, line: ~str) -> ~[ExpectedError] {
    let error_tag = ~"//~";
    let mut idx;
    match str::find_str(line, error_tag) {
      None => return ~[],
      Some(nn) => { idx = (nn as uint) + str::len(error_tag); }
    }

    // "//~^^^ kind msg" denotes a message expected
    // three lines above current line:
    let mut adjust_line = 0u;
    let len = str::len(line);
    while idx < len && line[idx] == ('^' as u8) {
        adjust_line += 1u;
        idx += 1u;
    }

    // Extract kind:
    while idx < len && line[idx] == (' ' as u8) { idx += 1u; }
    let start_kind = idx;
    while idx < len && line[idx] != (' ' as u8) { idx += 1u; }

    // FIXME: #4318 Instead of to_ascii and to_str_ascii, could use
    // to_ascii_consume and to_str_consume to not do a unnecessary copy.
    let kind = str::slice(line, start_kind, idx);
    let kind = kind.to_ascii().to_lower().to_str_ascii();

    // Extract msg:
    while idx < len && line[idx] == (' ' as u8) { idx += 1u; }
    let msg = str::slice(line, idx, len).to_owned();

    debug!("line=%u kind=%s msg=%s", line_num - adjust_line, kind, msg);

    return ~[ExpectedError{line: line_num - adjust_line, kind: kind,
                           msg: msg}];
}
