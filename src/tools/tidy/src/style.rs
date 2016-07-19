// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Tidy check to enforce various stylistic guidelines on the Rust codebase.
//!
//! Example checks are:
//!
//! * No lines over 100 characters
//! * No tabs
//! * No trailing whitespace
//! * No CR characters
//! * No `TODO` or `XXX` directives
//! * A valid license header is at the top
//!
//! A number of these checks can be opted-out of with various directives like
//! `// ignore-tidy-linelength`.

use std::fs::File;
use std::io::prelude::*;
use std::path::Path;

const COLS: usize = 100;
const LICENSE: &'static str = "\
Copyright <year> The Rust Project Developers. See the COPYRIGHT
file at the top-level directory of this distribution and at
http://rust-lang.org/COPYRIGHT.

Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
<LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
option. This file may not be copied, modified, or distributed
except according to those terms.";

pub fn check(path: &Path, bad: &mut bool) {
    let mut contents = String::new();
    super::walk(path, &mut super::filter_dirs, &mut |file| {
        let filename = file.file_name().unwrap().to_string_lossy();
        let extensions = [".rs", ".py", ".js", ".sh", ".c", ".h"];
        if extensions.iter().all(|e| !filename.ends_with(e)) ||
           filename.starts_with(".#") {
            return
        }
        if filename == "miniz.c" || filename.contains("jquery") {
            return
        }

        contents.truncate(0);
        t!(t!(File::open(file), file).read_to_string(&mut contents));
        let skip_cr = contents.contains("ignore-tidy-cr");
        let skip_tab = contents.contains("ignore-tidy-tab");
        let skip_length = contents.contains("ignore-tidy-linelength");
        for (i, line) in contents.split("\n").enumerate() {
            let mut err = |msg: &str| {
                println!("{}:{}: {}", file.display(), i + 1, msg);
                *bad = true;
            };
            if line.chars().count() > COLS && !skip_length {
                err(&format!("line longer than {} chars", COLS));
            }
            if line.contains("\t") && !skip_tab {
                err("tab character");
            }
            if line.ends_with(" ") || line.ends_with("\t") {
                err("trailing whitespace");
            }
            if line.contains("\r") && !skip_cr {
                err("CR character");
            }
            if filename != "style.rs" {
                if line.contains("TODO") {
                    err("TODO is deprecated; use FIXME")
                }
                if line.contains("//") && line.contains(" XXX") {
                    err("XXX is deprecated; use FIXME")
                }
            }
        }
        if !licenseck(file, &contents) {
            println!("{}: incorrect license", file.display());
            *bad = true;
        }
    })
}

fn licenseck(file: &Path, contents: &str) -> bool {
    if contents.contains("ignore-license") {
        return true
    }
    let exceptions = [
        "libstd/sync/mpsc/mpsc_queue.rs",
        "libstd/sync/mpsc/spsc_queue.rs",
    ];
    if exceptions.iter().any(|f| file.ends_with(f)) {
        return true
    }

    // Skip the BOM if it's there
    let bom = "\u{feff}";
    let contents = if contents.starts_with(bom) {&contents[3..]} else {contents};

    // See if the license shows up in the first 100 lines
    let lines = contents.lines().take(100).collect::<Vec<_>>();
    lines.windows(LICENSE.lines().count()).any(|window| {
        let offset = if window.iter().all(|w| w.starts_with("//")) {
            2
        } else if window.iter().all(|w| w.starts_with("#")) {
            1
        } else {
            return false
        };
        window.iter().map(|a| a[offset..].trim())
              .zip(LICENSE.lines()).all(|(a, b)| {
            a == b || match b.find("<year>") {
                Some(i) => a.starts_with(&b[..i]) && a.ends_with(&b[i+6..]),
                None => false,
            }
        })
    })

}
