// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
use rustfmt_diff::{Mismatch, DiffLine};
use std::io::{self, Write, Read, stdout};
use config::WriteMode;


pub fn output_heading(mode: WriteMode) -> Result<(), io::Error> {
    let stdout = stdout();
    let mut stdout = stdout.lock();
    if mode == WriteMode::Checkstyle {
        let mut xml_heading = String::new();
        xml_heading.push_str("<?xml version=\"1.0\" encoding=\"utf-8\"?>");
        xml_heading.push_str("\n");
        xml_heading.push_str("<checkstyle version=\"4.3\">");
        try!(write!(stdout, "{}", xml_heading));
    }
    Ok(())
}

pub fn output_footing(mode: WriteMode) -> Result<(), io::Error> {
    let stdout = stdout();
    let mut stdout = stdout.lock();
    if mode == WriteMode::Checkstyle {
        let mut xml_tail = String::new();
        xml_tail.push_str("</checkstyle>");
        try!(write!(stdout, "{}", xml_tail));
    }
    Ok(())
}

pub fn output_checkstyle_file<T>(mut writer: T,
                                 filename: &str,
                                 diff: Vec<Mismatch>)
                                 -> Result<(), io::Error>
    where T: Write
{
    try!(write!(writer, "<file name=\"{}\">", filename));
    for mismatch in diff {
        for line in mismatch.lines {
            match line {
                DiffLine::Expected(ref str) => {
                    let message = xml_escape_str(&str);
                    try!(write!(writer,
                                "<error line=\"{}\" severity=\"warning\" message=\"Should be \
                                 `{}`\" />",
                                mismatch.line_number,
                                message));
                }
                _ => {
                    // Do nothing with context and expected.
                }
            }
        }
    }
    try!(write!(writer, "</file>"));
    Ok(())
}

// Convert special characters into XML entities.
// This is needed for checkstyle output.
fn xml_escape_str(string: &str) -> String {
    let mut out = String::new();
    for c in string.chars() {
        match c {
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            '&' => out.push_str("&amp;"),
            _ => out.push(c),
        }
    }
    out
}
