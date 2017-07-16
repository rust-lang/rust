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
//! * No unexplained ` ```ignore ` or ` ```rust,ignore ` doc tests
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

const UNEXPLAINED_IGNORE_DOCTEST_INFO: &str = r#"unexplained "```ignore" doctest; try one:

* make the test actually pass, by adding necessary imports and declarations, or
* use "```text", if the code is not Rust code, or
* use "```compile_fail,Ennnn", if the code is expected to fail at compile time, or
* use "```should_panic", if the code is expected to fail at run time, or
* use "```no_run", if the code should type-check but not necessary linkable/runnable, or
* explain it like "```ignore (cannot-test-this-because-xxxx)", if the annotation cannot be avoided.

"#;

/// Parser states for line_is_url.
#[derive(PartialEq)]
#[allow(non_camel_case_types)]
enum LIUState { EXP_COMMENT_START,
                EXP_LINK_LABEL_OR_URL,
                EXP_URL,
                EXP_END }

/// True if LINE appears to be a line comment containing an URL,
/// possibly with a Markdown link label in front, and nothing else.
/// The Markdown link label, if present, may not contain whitespace.
/// Lines of this form are allowed to be overlength, because Markdown
/// offers no way to split a line in the middle of a URL, and the lengths
/// of URLs to external references are beyond our control.
fn line_is_url(line: &str) -> bool {
    use self::LIUState::*;
    let mut state: LIUState = EXP_COMMENT_START;

    for tok in line.split_whitespace() {
        match (state, tok) {
            (EXP_COMMENT_START, "//") => state = EXP_LINK_LABEL_OR_URL,
            (EXP_COMMENT_START, "///") => state = EXP_LINK_LABEL_OR_URL,
            (EXP_COMMENT_START, "//!") => state = EXP_LINK_LABEL_OR_URL,

            (EXP_LINK_LABEL_OR_URL, w)
                if w.len() >= 4 && w.starts_with("[") && w.ends_with("]:")
                => state = EXP_URL,

            (EXP_LINK_LABEL_OR_URL, w)
                if w.starts_with("http://") || w.starts_with("https://")
                => state = EXP_END,

            (EXP_URL, w)
                if w.starts_with("http://") || w.starts_with("https://")
                => state = EXP_END,

            (_, _) => return false,
        }
    }

    state == EXP_END
}

/// True if LINE is allowed to be longer than the normal limit.
/// Currently there is only one exception, for long URLs, but more
/// may be added in the future.
fn long_line_is_ok(line: &str) -> bool {
    if line_is_url(line) {
        return true;
    }

    false
}

pub fn check(path: &Path, bad: &mut bool) {
    let mut contents = String::new();
    super::walk(path, &mut super::filter_dirs, &mut |file| {
        let filename = file.file_name().unwrap().to_string_lossy();
        let extensions = [".rs", ".py", ".js", ".sh", ".c", ".h"];
        if extensions.iter().all(|e| !filename.ends_with(e)) ||
           filename.starts_with(".#") {
            return
        }

        contents.truncate(0);
        t!(t!(File::open(file), file).read_to_string(&mut contents));

        if contents.is_empty() {
            tidy_error!(bad, "{}: empty file", file.display());
        }

        let skip_cr = contents.contains("ignore-tidy-cr");
        let skip_tab = contents.contains("ignore-tidy-tab");
        let skip_length = contents.contains("ignore-tidy-linelength");
        let skip_end_whitespace = contents.contains("ignore-tidy-end-whitespace");
        for (i, line) in contents.split("\n").enumerate() {
            let mut err = |msg: &str| {
                tidy_error!(bad, "{}:{}: {}", file.display(), i + 1, msg);
            };
            if !skip_length && line.chars().count() > COLS
                && !long_line_is_ok(line) {
                    err(&format!("line longer than {} chars", COLS));
            }
            if line.contains("\t") && !skip_tab {
                err("tab character");
            }
            if !skip_end_whitespace && (line.ends_with(" ") || line.ends_with("\t")) {
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
            if line.ends_with("```ignore") || line.ends_with("```rust,ignore") {
                err(UNEXPLAINED_IGNORE_DOCTEST_INFO);
            }
        }
        if !licenseck(file, &contents) {
            tidy_error!(bad, "{}: incorrect license", file.display());
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
        } else if window.iter().all(|w| w.starts_with('#')) {
            1
        } else if window.iter().all(|w| w.starts_with(" *")) {
            2
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
