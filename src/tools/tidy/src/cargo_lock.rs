// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::path::Path;
use std::ffi::OsStr;

const CARGO_LOCK: &'static str = "Cargo.lock";

pub fn check(path: &Path, bad: &mut bool) {
    use std::process::Command;

    super::walk(path,
                &mut |path| super::filter_dirs(path) || path.ends_with("src/test"),
                &mut |file| {
        if let Some(CARGO_LOCK) = file.file_name().and_then(OsStr::to_str) {
            let rel_path = file.strip_prefix(path).unwrap();
            let git_friendly_path = rel_path.to_str().unwrap().replace("\\", "/");
            let ret_code = Command::new("git")
                                        .arg("diff")
                                        .arg("--exit-code")
                                        .arg("--patch")
                                        .arg("HEAD")
                                        .arg(&git_friendly_path)
                                        .current_dir(path)
                                        .status()
                                        .unwrap_or_else(|e| {
                                            panic!("could not run git diff-index: {}", e);
                                        });
            if !ret_code.success() {
                let parent_path = file.parent().unwrap().join("Cargo.toml");
                print!("dirty lock file found at {} ", rel_path.display());
                println!("please commit your changes or update the lock file by running:");
                println!("\n\tcargo update --manifest-path {}", parent_path.display());
                *bad = true;
            }
        }
    });
}
