// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


// TODO: add tests

use strings::string_buffer::StringBuffer;

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{self, Write, Read, stdout, BufWriter};

use config::{NewlineStyle, Config, WriteMode};
use rustfmt_diff::{make_diff, print_diff, Mismatch, DiffLine};

// A map of the files of a crate, with their new content
pub type FileMap = HashMap<String, StringBuffer>;

// Append a newline to the end of each file.
pub fn append_newlines(file_map: &mut FileMap) {
    for (_, s) in file_map.iter_mut() {
        s.push_str("\n");
    }
}

pub fn write_all_files(file_map: &FileMap,
                       mode: WriteMode,
                       config: &Config)
                       -> Result<(), io::Error> {
    output_heading(mode).ok();
    for filename in file_map.keys() {
        try!(write_file(&file_map[filename], filename, mode, config));
    }
    output_trailing(mode).ok();

    Ok(())
}

pub fn output_heading(mode: WriteMode) -> Result<(), io::Error> {
    let stdout = stdout();
    let mut stdout = stdout.lock();
    match mode {
        WriteMode::Checkstyle => {
            let mut xml_heading = String::new();
            xml_heading.push_str("<?xml version=\"1.0\" encoding=\"utf-8\"?>");
            xml_heading.push_str("\n");
            xml_heading.push_str("<checkstyle version=\"4.3\">");
            try!(write!(stdout, "{}", xml_heading));
            Ok(())
        }
        _ => Ok(()),
    }
}

pub fn output_trailing(mode: WriteMode) -> Result<(), io::Error> {
    let stdout = stdout();
    let mut stdout = stdout.lock();
    match mode {
        WriteMode::Checkstyle => {
            let mut xml_tail = String::new();
            xml_tail.push_str("</checkstyle>");
            try!(write!(stdout, "{}", xml_tail));
            Ok(())
        }
        _ => Ok(()),
    }
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
                    // TODO XML encode str here.
                    try!(write!(writer,
                                "<error line=\"{}\" severity=\"error\" message=\"Should be \
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

// Prints all newlines either as `\n` or as `\r\n`.
pub fn write_system_newlines<T>(writer: T,
                                text: &StringBuffer,
                                config: &Config)
                                -> Result<(), io::Error>
    where T: Write
{
    // Buffer output, since we're writing a since char at a time.
    let mut writer = BufWriter::new(writer);

    let style = if config.newline_style == NewlineStyle::Native {
        if cfg!(windows) {
            NewlineStyle::Windows
        } else {
            NewlineStyle::Unix
        }
    } else {
        config.newline_style
    };

    match style {
        NewlineStyle::Unix => write!(writer, "{}", text),
        NewlineStyle::Windows => {
            for (c, _) in text.chars() {
                match c {
                    '\n' => try!(write!(writer, "\r\n")),
                    '\r' => continue,
                    c => try!(write!(writer, "{}", c)),
                }
            }
            Ok(())
        }
        NewlineStyle::Native => unreachable!(),
    }
}

pub fn write_file(text: &StringBuffer,
                  filename: &str,
                  mode: WriteMode,
                  config: &Config)
                  -> Result<Option<String>, io::Error> {

    fn source_and_formatted_text(text: &StringBuffer,
                                 filename: &str,
                                 config: &Config)
                                 -> Result<(String, String), io::Error> {
        let mut f = try!(File::open(filename));
        let mut ori_text = String::new();
        try!(f.read_to_string(&mut ori_text));
        let mut v = Vec::new();
        try!(write_system_newlines(&mut v, text, config));
        let fmt_text = String::from_utf8(v).unwrap();
        Ok((ori_text, fmt_text))
    }

    match mode {
        WriteMode::Replace => {
            if let Ok((ori, fmt)) = source_and_formatted_text(text, filename, config) {
                if fmt != ori {
                    // Do a little dance to make writing safer - write to a temp file
                    // rename the original to a .bk, then rename the temp file to the
                    // original.
                    let tmp_name = filename.to_owned() + ".tmp";
                    let bk_name = filename.to_owned() + ".bk";
                    {
                        // Write text to temp file
                        let tmp_file = try!(File::create(&tmp_name));
                        try!(write_system_newlines(tmp_file, text, config));
                    }

                    try!(fs::rename(filename, bk_name));
                    try!(fs::rename(tmp_name, filename));
                }
            }
        }
        WriteMode::Overwrite => {
            // Write text directly over original file.
            let file = try!(File::create(filename));
            try!(write_system_newlines(file, text, config));
        }
        WriteMode::Plain => {
            let stdout = stdout();
            let stdout = stdout.lock();
            try!(write_system_newlines(stdout, text, config));
        }
        WriteMode::Display | WriteMode::Coverage => {
            println!("{}:\n", filename);
            let stdout = stdout();
            let stdout = stdout.lock();
            try!(write_system_newlines(stdout, text, config));
        }
        WriteMode::Diff => {
            println!("Diff of {}:\n", filename);
            if let Ok((ori, fmt)) = source_and_formatted_text(text, filename, config) {
                print_diff(make_diff(&ori, &fmt, 3),
                           |line_num| format!("\nDiff at line {}:", line_num));
            }
        }
        WriteMode::Default => {
            unreachable!("The WriteMode should NEVER Be default at this point!");
        }
        WriteMode::Checkstyle => {
            let stdout = stdout();
            let stdout = stdout.lock();
            // Generate the diff for the current file.
            let mut f = try!(File::open(filename));
            let mut ori_text = String::new();
            try!(f.read_to_string(&mut ori_text));
            let mut v = Vec::new();
            try!(write_system_newlines(&mut v, text, config));
            let fmt_text = String::from_utf8(v).unwrap();
            let diff = make_diff(&ori_text, &fmt_text, 3);
            // Output the XML tags for the lines that are different.
            output_checkstyle_file(stdout, filename, diff).unwrap();
        }
        WriteMode::Return => {
            // io::Write is not implemented for String, working around with
            // Vec<u8>
            let mut v = Vec::new();
            try!(write_system_newlines(&mut v, text, config));
            // won't panic, we are writing correct utf8
            return Ok(Some(String::from_utf8(v).unwrap()));
        }
    }

    Ok(None)
}
