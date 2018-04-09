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

use std::path::Path;

// See rust-lang/rust#48879: In addition to the mapping from `foo.rs`
// to `foo.stderr`/`foo.stdout`, we also can optionally have
// `foo.$mode.stderr`, where $mode is one of the strings on this list,
// as an alternative to use when running under that mode.
static COMPARE_MODE_NAMES: [&'static str; 1] = ["nll"];

pub fn check(path: &Path, bad: &mut bool) {
    super::walk_many(&[&path.join("test/ui"), &path.join("test/ui-fulldeps")],
                     &mut |_| false,
                     &mut |file_path| {
        if let Some(ext) = file_path.extension() {
            if (ext == "stderr" || ext == "stdout") && !file_path.with_extension("rs").exists() {

                // rust-lang/rust#48879: this fn used to be beautful
                // because Path API special-cases replacing
                // extensions. That works great for ".stderr" but not
                // so well for ".nll.stderr". To support the latter,
                // we explicitly search backwards for mode's starting
                // point and build corresponding source name.
                let filename = file_path.file_name().expect("need filename")
                    .to_str().expect("need UTF-8 filename");
                let found_matching_prefix = COMPARE_MODE_NAMES.iter().any(|mode| {
                    if let Some(r_idx) = filename.rfind(&format!(".{}", mode)) {
                        let source_name = format!("{}.rs", &filename[0..r_idx]);
                        let source_path = file_path.with_file_name(source_name);
                        source_path.exists()
                    } else {
                        false
                    }
                });

                if !found_matching_prefix {
                    println!("Stray file with UI testing output: {:?}", file_path);
                    *bad = true;
                }
            }
        }
    });
}
