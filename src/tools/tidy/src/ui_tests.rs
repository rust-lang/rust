//! Tidy check to ensure below in UI test directories:
//! - the number of entries in each directory must be less than `ENTRY_LIMIT`
//! - there are no stray `.stderr` files

use ignore::Walk;
use ignore::WalkBuilder;
use std::fs;
use std::path::Path;

const ENTRY_LIMIT: usize = 1000;
// FIXME: The following limits should be reduced eventually.
const ROOT_ENTRY_LIMIT: usize = 940;
const ISSUES_ENTRY_LIMIT: usize = 1978;

fn check_entries(path: &Path, bad: &mut bool) {
    for dir in Walk::new(&path.join("ui")) {
        if let Ok(entry) = dir {
            if entry.file_type().map(|ft| ft.is_dir()).unwrap_or(false) {
                let dir_path = entry.path();
                // Use special values for these dirs.
                let is_root = path.join("ui") == dir_path;
                let is_issues_dir = path.join("ui/issues") == dir_path;
                let limit = if is_root {
                    ROOT_ENTRY_LIMIT
                } else if is_issues_dir {
                    ISSUES_ENTRY_LIMIT
                } else {
                    ENTRY_LIMIT
                };

                let count = WalkBuilder::new(&dir_path)
                    .max_depth(Some(1))
                    .build()
                    .into_iter()
                    .collect::<Vec<_>>()
                    .len()
                    - 1; // remove the dir itself

                if count > limit {
                    tidy_error!(
                        bad,
                        "following path contains more than {} entries, \
                            you should move the test to some relevant subdirectory (current: {}): {}",
                        limit,
                        count,
                        dir_path.display()
                    );
                }
            }
        }
    }
}

pub fn check(path: &Path, bad: &mut bool) {
    check_entries(&path, bad);
    for path in &[&path.join("ui"), &path.join("ui-fulldeps")] {
        crate::walk::walk_no_read(path, |_| false, &mut |entry| {
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
                        tidy_error!(bad, "Stray file with UI testing output: {:?}", file_path);
                    }

                    if let Ok(metadata) = fs::metadata(file_path) {
                        if metadata.len() == 0 {
                            tidy_error!(bad, "Empty file with UI testing output: {:?}", file_path);
                        }
                    }
                }
            }
        });
    }
}
