// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::fs;
use std::io::{self, Write};

use checkstyle::output_checkstyle_file;
use config::{Config, EmitMode, FileName, Verbosity};
use rustfmt_diff::{make_diff, output_modified, print_diff};

#[cfg(test)]
use formatting::FileRecord;

// Append a newline to the end of each file.
pub fn append_newline(s: &mut String) {
    s.push_str("\n");
}

#[cfg(test)]
pub(crate) fn write_all_files<T>(
    source_file: &[FileRecord],
    out: &mut T,
    config: &Config,
) -> Result<(), io::Error>
where
    T: Write,
{
    if config.emit_mode() == EmitMode::Checkstyle {
        write!(out, "{}", ::checkstyle::header())?;
    }
    for &(ref filename, ref text) in source_file {
        write_file(text, filename, out, config)?;
    }
    if config.emit_mode() == EmitMode::Checkstyle {
        write!(out, "{}", ::checkstyle::footer())?;
    }

    Ok(())
}

pub fn write_file<T>(
    formatted_text: &str,
    filename: &FileName,
    out: &mut T,
    config: &Config,
) -> Result<bool, io::Error>
where
    T: Write,
{
    let filename_to_path = || match *filename {
        FileName::Real(ref path) => path,
        _ => panic!("cannot format `{}` and emit to files", filename),
    };

    match config.emit_mode() {
        EmitMode::Files if config.make_backup() => {
            let filename = filename_to_path();
            let ori = fs::read_to_string(filename)?;
            if ori != formatted_text {
                // Do a little dance to make writing safer - write to a temp file
                // rename the original to a .bk, then rename the temp file to the
                // original.
                let tmp_name = filename.with_extension("tmp");
                let bk_name = filename.with_extension("bk");

                fs::write(&tmp_name, formatted_text)?;
                fs::rename(filename, bk_name)?;
                fs::rename(tmp_name, filename)?;
            }
        }
        EmitMode::Files => {
            // Write text directly over original file if there is a diff.
            let filename = filename_to_path();
            let ori = fs::read_to_string(filename)?;
            if ori != formatted_text {
                fs::write(filename, formatted_text)?;
            }
        }
        EmitMode::Stdout | EmitMode::Coverage => {
            if config.verbose() != Verbosity::Quiet {
                println!("{}:\n", filename);
            }
            write!(out, "{}", formatted_text)?;
        }
        EmitMode::ModifiedLines => {
            let filename = filename_to_path();
            let ori = fs::read_to_string(filename)?;
            let mismatch = make_diff(&ori, formatted_text, 0);
            let has_diff = !mismatch.is_empty();
            output_modified(out, mismatch);
            return Ok(has_diff);
        }
        EmitMode::Checkstyle => {
            let filename = filename_to_path();
            let ori = fs::read_to_string(filename)?;
            let diff = make_diff(&ori, formatted_text, 3);
            output_checkstyle_file(out, filename, diff)?;
        }
        EmitMode::Diff => {
            let filename = filename_to_path();
            let ori = fs::read_to_string(filename)?;
            let mismatch = make_diff(&ori, formatted_text, 3);
            let has_diff = !mismatch.is_empty();
            print_diff(
                mismatch,
                |line_num| format!("Diff in {} at line {}:", filename.display(), line_num),
                config,
            );
            return Ok(has_diff);
        }
    }

    // when we are not in diff mode, don't indicate differing files
    Ok(false)
}
