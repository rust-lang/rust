// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Tidy check to verify the validity of long error diagnostic codes.
//!
//! This ensures that error codes are used at most once and also prints out some
//! statistics about the error codes.

use std::collections::HashMap;
use std::fs::File;
use std::io::prelude::*;
use std::path::Path;

pub fn check(path: &Path, bad: &mut bool) {
    let mut contents = String::new();
    let mut map = HashMap::new();
    super::walk(path,
                &mut |path| super::filter_dirs(path) || path.ends_with("src/test"),
                &mut |file| {
        let filename = file.file_name().unwrap().to_string_lossy();
        if filename != "diagnostics.rs" && filename != "diagnostic_list.rs" {
            return
        }

        contents.truncate(0);
        t!(t!(File::open(file)).read_to_string(&mut contents));

        // In the register_long_diagnostics! macro, entries look like this:
        //
        // EXXXX: r##"
        // <Long diagnostic message>
        // "##,
        //
        // and these long messages often have error codes themselves inside
        // them, but we don't want to report duplicates in these cases. This
        // variable keeps track of whether we're currently inside one of these
        // long diagnostic messages.
        let mut inside_long_diag = false;
        for (num, line) in contents.lines().enumerate() {
            if inside_long_diag {
                inside_long_diag = !line.contains("\"##");
                continue
            }

            let mut search = line;
            while let Some(i) = search.find("E") {
                search = &search[i + 1..];
                let code = if search.len() > 4 {
                    search[..4].parse::<u32>()
                } else {
                    continue
                };
                let code = match code {
                    Ok(n) => n,
                    Err(..) => continue,
                };
                map.entry(code).or_insert(Vec::new())
                   .push((file.to_owned(), num + 1, line.to_owned()));
                break
            }

            inside_long_diag = line.contains("r##\"");
        }
    });

    let mut max = 0;
    for (&code, entries) in map.iter() {
        if code > max {
            max = code;
        }
        if entries.len() == 1 {
            continue
        }

        println!("duplicate error code: {}", code);
        for &(ref file, line_num, ref line) in entries.iter() {
            println!("{}:{}: {}", file.display(), line_num, line);
        }
        *bad = true;
    }

    if !*bad {
        println!("* {} error codes", map.len());
        println!("* highest error code: E{:04}", max);
    }
}
