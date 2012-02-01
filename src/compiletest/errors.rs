import std::io;
import io::reader_util;
import std::fs;

import common::config;

export load_errors;
export expected_error;

type expected_error = { line: uint, kind: str, msg: str };

// Load any test directives embedded in the file
fn load_errors(testfile: str) -> [expected_error] {
    let error_patterns = [];
    let rdr = result::get(io::file_reader(testfile));
    let line_num = 1u;
    while !rdr.eof() {
        let ln = rdr.read_line();
        error_patterns += parse_expected(line_num, ln);
        line_num += 1u;
    }
    ret error_patterns;
}

fn parse_expected(line_num: uint, line: str) -> [expected_error] unsafe {
    let error_tag = "//!";
    let idx0 = str::find(line, error_tag);
    if idx0 < 0 { ret []; }
    let idx = (idx0 as uint) + str::byte_len(error_tag);

    // "//!^^^ kind msg" denotes a message expected
    // three lines above current line:
    let adjust_line = 0u;
    let len = str::byte_len(line);
    while idx < len && line[idx] == ('^' as u8) {
        adjust_line += 1u;
        idx += 1u;
    }

    // Extract kind:
    while idx < len && line[idx] == (' ' as u8) { idx += 1u; }
    let start_kind = idx;
    while idx < len && line[idx] != (' ' as u8) { idx += 1u; }
    let kind = str::to_lower(str::unsafe::slice(line, start_kind, idx));

    // Extract msg:
    while idx < len && line[idx] == (' ' as u8) { idx += 1u; }
    let msg = str::unsafe::slice(line, idx, len);

    #debug("line=%u kind=%s msg=%s", line_num - adjust_line, kind, msg);

    ret [{line: line_num - adjust_line, kind: kind, msg: msg}];
}
