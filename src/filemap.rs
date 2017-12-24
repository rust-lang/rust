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

use std::fs::{self, File};
use std::io::{self, BufWriter, Read, Write};
use std::path::Path;

use checkstyle::{output_checkstyle_file, output_footer, output_header};
use config::{Config, NewlineStyle, WriteMode};
use rustfmt_diff::{make_diff, print_diff, Mismatch};
use syntax::codemap::FileName;

// A map of the files of a crate, with their new content
pub type FileMap = Vec<FileRecord>;

pub type FileRecord = (FileName, String);

// Append a newline to the end of each file.
pub fn append_newline(s: &mut String) {
    s.push_str("\n");
}

pub fn write_all_files<T>(
    file_map: &[FileRecord],
    out: &mut T,
    config: &Config,
) -> Result<(), io::Error>
where
    T: Write,
{
    output_header(out, config.write_mode()).ok();
    for &(ref filename, ref text) in file_map {
        write_file(text, filename, out, config)?;
    }
    output_footer(out, config.write_mode()).ok();

    Ok(())
}

// Prints all newlines either as `\n` or as `\r\n`.
pub fn write_system_newlines<T>(writer: T, text: &str, config: &Config) -> Result<(), io::Error>
where
    T: Write,
{
    // Buffer output, since we're writing a since char at a time.
    let mut writer = BufWriter::new(writer);

    let style = if config.newline_style() == NewlineStyle::Native {
        if cfg!(windows) {
            NewlineStyle::Windows
        } else {
            NewlineStyle::Unix
        }
    } else {
        config.newline_style()
    };

    match style {
        NewlineStyle::Unix => write!(writer, "{}", text),
        NewlineStyle::Windows => {
            for c in text.chars() {
                match c {
                    '\n' => write!(writer, "\r\n")?,
                    '\r' => continue,
                    c => write!(writer, "{}", c)?,
                }
            }
            Ok(())
        }
        NewlineStyle::Native => unreachable!(),
    }
}

pub fn write_file<T>(
    text: &str,
    filename: &FileName,
    out: &mut T,
    config: &Config,
) -> Result<bool, io::Error>
where
    T: Write,
{
    fn source_and_formatted_text(
        text: &str,
        filename: &Path,
        config: &Config,
    ) -> Result<(String, String), io::Error> {
        let mut f = File::open(filename)?;
        let mut ori_text = String::new();
        f.read_to_string(&mut ori_text)?;
        let mut v = Vec::new();
        write_system_newlines(&mut v, text, config)?;
        let fmt_text = String::from_utf8(v).unwrap();
        Ok((ori_text, fmt_text))
    }

    fn create_diff(
        filename: &Path,
        text: &str,
        config: &Config,
    ) -> Result<Vec<Mismatch>, io::Error> {
        let (ori, fmt) = source_and_formatted_text(text, filename, config)?;
        Ok(make_diff(&ori, &fmt, 3))
    }

    let filename_to_path = || match *filename {
        FileName::Real(ref path) => path,
        _ => panic!("cannot format `{}` with WriteMode::Replace", filename),
    };

    match config.write_mode() {
        WriteMode::Replace => {
            let filename = filename_to_path();
            if let Ok((ori, fmt)) = source_and_formatted_text(text, filename, config) {
                if fmt != ori {
                    // Do a little dance to make writing safer - write to a temp file
                    // rename the original to a .bk, then rename the temp file to the
                    // original.
                    let tmp_name = filename.with_extension("tmp");
                    let bk_name = filename.with_extension("bk");
                    {
                        // Write text to temp file
                        let tmp_file = File::create(&tmp_name)?;
                        write_system_newlines(tmp_file, text, config)?;
                    }

                    fs::rename(filename, bk_name)?;
                    fs::rename(tmp_name, filename)?;
                }
            }
        }
        WriteMode::Overwrite => {
            // Write text directly over original file if there is a diff.
            let filename = filename_to_path();
            let (source, formatted) = source_and_formatted_text(text, filename, config)?;
            if source != formatted {
                let file = File::create(filename)?;
                write_system_newlines(file, text, config)?;
            }
        }
        WriteMode::Plain => {
            write_system_newlines(out, text, config)?;
        }
        WriteMode::Display | WriteMode::Coverage => {
            println!("{}:\n", filename);
            write_system_newlines(out, text, config)?;
        }
        WriteMode::Diff => {
            let filename = filename_to_path();
            if let Ok((ori, fmt)) = source_and_formatted_text(text, filename, config) {
                let mismatch = make_diff(&ori, &fmt, 3);
                let has_diff = !mismatch.is_empty();
                print_diff(
                    mismatch,
                    |line_num| format!("Diff in {} at line {}:", filename.display(), line_num),
                    config.color(),
                );
                return Ok(has_diff);
            }
        }
        WriteMode::Checkstyle => {
            let filename = filename_to_path();
            let diff = create_diff(filename, text, config)?;
            output_checkstyle_file(out, filename, diff)?;
        }
    }

    // when we are not in diff mode, don't indicate differing files
    Ok(false)
}
