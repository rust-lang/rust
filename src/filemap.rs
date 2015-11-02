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
use std::io::{self, Write, Read, stdout};

use WriteMode;
use config::{NewlineStyle, Config};
use rustfmt_diff::{make_diff, print_diff};

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
                       -> Result<(HashMap<String, String>), io::Error> {
    let mut result = HashMap::new();
    for filename in file_map.keys() {
        let one_result = try!(write_file(&file_map[filename], filename, mode, config));
        if let Some(r) = one_result {
            result.insert(filename.clone(), r);
        }
    }

    Ok(result)
}

pub fn write_file(text: &StringBuffer,
                  filename: &str,
                  mode: WriteMode,
                  config: &Config)
                  -> Result<Option<String>, io::Error> {

    // prints all newlines either as `\n` or as `\r\n`
    fn write_system_newlines<T>(mut writer: T,
                                text: &StringBuffer,
                                config: &Config)
                                -> Result<(), io::Error>
        where T: Write
    {
        match config.newline_style {
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
        }
    }

    match mode {
        WriteMode::Replace => {
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
        WriteMode::Overwrite => {
            // Write text directly over original file.
            let file = try!(File::create(filename));
            try!(write_system_newlines(file, text, config));
        }
        WriteMode::NewFile(extn) => {
            let filename = filename.to_owned() + "." + extn;
            let file = try!(File::create(&filename));
            try!(write_system_newlines(file, text, config));
        }
        WriteMode::Plain => {
            let stdout = stdout();
            let stdout_lock = stdout.lock();
            try!(write_system_newlines(stdout_lock, text, config));
        }
        WriteMode::Display | WriteMode::Coverage => {
            println!("{}:\n", filename);
            let stdout = stdout();
            let stdout_lock = stdout.lock();
            try!(write_system_newlines(stdout_lock, text, config));
        }
        WriteMode::Diff => {
            println!("Diff of {}:\n", filename);
            let mut f = try!(File::open(filename));
            let mut ori_text = String::new();
            try!(f.read_to_string(&mut ori_text));
            let mut v = Vec::new();
            try!(write_system_newlines(&mut v, text, config));
            let fmt_text = String::from_utf8(v).unwrap();
            let diff = make_diff(&ori_text, &fmt_text, 3);
            print_diff(diff, |line_num| format!("\nDiff at line {}:", line_num));
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
