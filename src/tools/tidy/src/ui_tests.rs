// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Tidy check to ensure that there are no stray `.stderr` files in UI test directories.

use std::fs;
use std::path::Path;

pub fn check(path: &Path, bad: &mut bool) {
    super::walk_many(
        &[&path.join("test/ui"), &path.join("test/ui-fulldeps")],
        &mut |_| false,
        &mut |file_path| {
            if let Some(ext) = file_path.extension() {
                if ext == "stderr" || ext == "stdout" {
                    // Test output filenames have the format:
                    // $testname.stderr
                    // $testname.$mode.stderr
                    // $testname.$revision.stderr
                    // $testname.$revision.$mode.stderr
                    //
                    // For now, just make sure that there is a corresponding
                    // $testname.rs file.
                    let testname = file_path
                        .file_name()
                        .unwrap()
                        .to_str()
                        .unwrap()
                        .splitn(2, '.')
                        .next()
                        .unwrap();
                    if !file_path
                        .with_file_name(testname)
                        .with_extension("rs")
                        .exists()
                    {
                        println!("Stray file with UI testing output: {:?}", file_path);
                        *bad = true;
                    }

                    if let Ok(metadata) = fs::metadata(file_path) {
                        if metadata.len() == 0 {
                            println!("Empty file with UI testing output: {:?}", file_path);
                            *bad = true;
                        }
                    }
                }
            }
        },
    );
}
