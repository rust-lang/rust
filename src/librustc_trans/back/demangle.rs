// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::fmt::Write as fmt_Write;
use std::io;
use std::io::Read;
use std::io::Write as io_Write;
use std::fs;
use std::path::Path;

use rustc_demangle;

// find raw symbol in line
fn find_symbol_in_line(mut line: &str) -> Option<&str> {
    while !line.is_empty() {
        // Skip underscores, because on macOS compiler adds extra underscore,
        // and rustc_demangle successfully demangles symbols without leading underscore.
        let zn = match line.find("ZN") {
            Some(pos) => pos,
            None => return None,
        };
        line = &line[zn..];

        let mut pos = "ZN".len();

        fn is_valid_char_in_symbol(c: u8) -> bool {
            // symbol_names::sanitize contains complete list of valid chars
            match c {
                b'$' | b'.' | b'_' => true,
                b'a' ... b'z' => true,
                b'A' ... b'Z' => true,
                b'0' ... b'9' => true,
                _ => false,
            }
        }

        while pos != line.len() && is_valid_char_in_symbol(line.as_bytes()[pos]) {
            pos += 1;
        }

        if line.as_bytes()[pos - 1] == b'E' {
            return Some(&line[..pos]);
        }

        line = &line[pos..];
    }

    None
}

fn demangle_asm_in_text(text: &str) -> String {
    let mut r = String::new();
    for line in text.lines() {
        // Do not comment comments.
        if !line.starts_with(";") {
            // Find a symbol, demangle at most one symbol on line.
            if let Some(sym) = find_symbol_in_line(line) {
                // If couldn't demangle, probably it is false positive in `find_symbol_in_line`.
                if let Ok(dem) = rustc_demangle::try_demangle(sym) {
                    let mut start = String::new();
                    if line.starts_with("\t") {
                        start.push_str(";\t");
                    } else {
                        // If line is indented with spaces, keep indentation.
                        let leading_spaces = line.chars().filter(|&c| c == ' ').count();
                        start.push_str("; ");
                        while start.len() < leading_spaces {
                            start.push(' ');
                        }
                    }
                    write!(r, "{}{:#}\n", start, dem).unwrap();
                }
            }
        }
        r.push_str(line);
        r.push_str("\n");
    }
    r
}

// Only error is IO error.
pub fn demangle_asm_in_file_in_place(path: &Path) -> io::Result<()> {
    let mut file = fs::File::open(path)?;
    let mut text = String::new();
    file.read_to_string(&mut text)?;
    drop(file); // close

    let demangled = demangle_asm_in_text(&text);

    let mut file = fs::File::create(path)?;
    file.write_all(demangled.as_bytes())?;
    Ok(())
}
