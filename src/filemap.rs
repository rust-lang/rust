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

use std::fs::{self, File};
use std::io::{self, Write, Read, stdout, BufWriter};

use config::{NewlineStyle, Config, WriteMode};
use rustfmt_diff::{make_diff, print_diff, Mismatch};
use checkstyle::{output_header, output_footer, output_checkstyle_file};

// A map of the files of a crate, with their new content
pub type FileMap = Vec<FileRecord>;

pub type FileRecord = (String, StringBuffer);

// Append a newline to the end of each file.
pub fn append_newline(s: &mut StringBuffer) {
    s.push_str("\n");
}

pub fn write_all_files<T>(file_map: &FileMap, out: &mut T, config: &Config) -> Result<(), io::Error>
    where T: Write
{
    output_header(out, config.write_mode).ok();
    for &(ref filename, ref text) in file_map {
        try!(write_file(text, filename, out, config));
    }
    output_footer(out, config.write_mode).ok();

    Ok(())
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

pub fn write_file<T>(text: &StringBuffer,
                     filename: &str,
                     out: &mut T,
                     config: &Config)
                     -> Result<Option<String>, io::Error>
    where T: Write
{

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

    fn create_diff(filename: &str,
                   text: &StringBuffer,
                   config: &Config)
                   -> Result<Vec<Mismatch>, io::Error> {
        let (ori, fmt) = try!(source_and_formatted_text(text, filename, config));
        Ok(make_diff(&ori, &fmt, 3))
    }

    match config.write_mode {
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
        WriteMode::Checkstyle => {
            let diff = try!(create_diff(filename, text, config));
            try!(output_checkstyle_file(out, filename, diff));
        }
    }

    Ok(None)
}
