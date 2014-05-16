// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::io::{BufferedReader, File};

pub struct ExpectedError {
    pub line: uint,
    pub kind: StrBuf,
    pub msg: StrBuf,
}

// Load any test directives embedded in the file
pub fn load_errors(testfile: &Path) -> Vec<ExpectedError> {

    let mut error_patterns = Vec::new();
    let mut rdr = BufferedReader::new(File::open(testfile).unwrap());
    let mut line_num = 1u;
    for ln in rdr.lines() {
        error_patterns.push_all_move(parse_expected(line_num,
                                                    ln.unwrap().to_strbuf()));
        line_num += 1u;
    }
    return error_patterns;
}

fn parse_expected(line_num: uint, line: StrBuf) -> Vec<ExpectedError> {
    let line = line.as_slice().trim().to_strbuf();
    let error_tag = "//~".to_strbuf();
    let mut idx;
    match line.as_slice().find_str(error_tag.as_slice()) {
      None => return Vec::new(),
      Some(nn) => { idx = (nn as uint) + error_tag.len(); }
    }

    // "//~^^^ kind msg" denotes a message expected
    // three lines above current line:
    let mut adjust_line = 0u;
    let len = line.len();
    while idx < len && line.as_slice()[idx] == ('^' as u8) {
        adjust_line += 1u;
        idx += 1u;
    }

    // Extract kind:
    while idx < len && line.as_slice()[idx] == (' ' as u8) {
        idx += 1u;
    }
    let start_kind = idx;
    while idx < len && line.as_slice()[idx] != (' ' as u8) {
        idx += 1u;
    }

    let kind = line.as_slice().slice(start_kind, idx);
    let kind = kind.to_ascii().to_lower().into_str().to_strbuf();

    // Extract msg:
    while idx < len && line.as_slice()[idx] == (' ' as u8) {
        idx += 1u;
    }
    let msg = line.as_slice().slice(idx, len).to_strbuf();

    debug!("line={} kind={} msg={}", line_num - adjust_line, kind, msg);

    return vec!(ExpectedError{
        line: line_num - adjust_line,
        kind: kind,
        msg: msg,
    });
}
