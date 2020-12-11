//! Tidy check to ensure that there are no stray `.stderr` files in UI test directories.

use std::fs;
use std::path::Path;

pub fn check(path: &Path, bad: &mut bool) {
    for path in &[&path.join("test/ui"), &path.join("test/ui-fulldeps")] {
        super::walk_no_read(path, &mut |_| false, &mut |entry| {
            let file_path = entry.path();
            if let Some(ext) = file_path.extension() {
                if ext == "stderr" || ext == "stdout" {
                    // Test output filenames have one of the formats:
                    // ```
                    // $testname.stderr
                    // $testname.$mode.stderr
                    // $testname.$revision.stderr
                    // $testname.$revision.$mode.stderr
                    // ```
                    //
                    // For now, just make sure that there is a corresponding
                    // `$testname.rs` file.
                    //
                    // NB: We do not use file_stem() as some file names have multiple `.`s and we
                    // must strip all of them.
                    let testname =
                        file_path.file_name().unwrap().to_str().unwrap().split_once('.').unwrap().0;
                    if !file_path.with_file_name(testname).with_extension("rs").exists() {
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
        });
    }
}
