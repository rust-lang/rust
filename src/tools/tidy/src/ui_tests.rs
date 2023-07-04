//! Tidy check to ensure below in UI test directories:
//! - the number of entries in each directory must be less than `ENTRY_LIMIT`
//! - there are no stray `.stderr` files

use ignore::Walk;
use std::collections::HashMap;
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};

const ENTRY_LIMIT: usize = 900;
// FIXME: The following limits should be reduced eventually.
const ISSUES_ENTRY_LIMIT: usize = 1896;
const ROOT_ENTRY_LIMIT: usize = 870;

const EXPECTED_TEST_FILE_EXTENSIONS: &[&str] = &[
    "rs",     // test source files
    "stderr", // expected stderr file, corresponds to a rs file
    "stdout", // expected stdout file, corresponds to a rs file
    "fixed",  // expected source file after applying fixes
    "md",     // test directory descriptions
    "ftl",    // translation tests
];

const EXTENSION_EXCEPTION_PATHS: &[&str] = &[
    "tests/ui/asm/named-asm-labels.s", // loading an external asm file to test named labels lint
    "tests/ui/check-cfg/my-awesome-platform.json", // testing custom targets with cfgs
    "tests/ui/commandline-argfile-badutf8.args", // passing args via a file
    "tests/ui/commandline-argfile.args", // passing args via a file
    "tests/ui/crate-loading/auxiliary/libfoo.rlib", // testing loading a manually created rlib
    "tests/ui/include-macros/data.bin", // testing including data with the include macros
    "tests/ui/include-macros/file.txt", // testing including data with the include macros
    "tests/ui/macros/macro-expanded-include/file.txt", // testing including data with the include macros
    "tests/ui/macros/not-utf8.bin", // testing including data with the include macros
    "tests/ui/macros/syntax-extension-source-utils-files/includeme.fragment", // more include
    "tests/ui/unused-crate-deps/test.mk", // why would you use make
    "tests/ui/proc-macro/auxiliary/included-file.txt", // more include
    "tests/ui/invalid/foo.natvis.xml", // sample debugger visualizer
];

fn check_entries(tests_path: &Path, bad: &mut bool) {
    let mut directories: HashMap<PathBuf, usize> = HashMap::new();

    for dir in Walk::new(&tests_path.join("ui")) {
        if let Ok(entry) = dir {
            let parent = entry.path().parent().unwrap().to_path_buf();
            *directories.entry(parent).or_default() += 1;
        }
    }

    let (mut max, mut max_root, mut max_issues) = (0usize, 0usize, 0usize);
    for (dir_path, count) in directories {
        // Use special values for these dirs.
        let is_root = tests_path.join("ui") == dir_path;
        let is_issues_dir = tests_path.join("ui/issues") == dir_path;
        let (limit, maxcnt) = if is_root {
            (ROOT_ENTRY_LIMIT, &mut max_root)
        } else if is_issues_dir {
            (ISSUES_ENTRY_LIMIT, &mut max_issues)
        } else {
            (ENTRY_LIMIT, &mut max)
        };
        *maxcnt = (*maxcnt).max(count);
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
    if ROOT_ENTRY_LIMIT > max_root {
        tidy_error!(
            bad,
            "`ROOT_ENTRY_LIMIT` is too high (is {ROOT_ENTRY_LIMIT}, should be {max_root})"
        );
    }
    if ISSUES_ENTRY_LIMIT > max_issues {
        tidy_error!(
            bad,
            "`ISSUES_ENTRY_LIMIT` is too high (is {ISSUES_ENTRY_LIMIT}, should be {max_issues})"
        );
    }
}

pub fn check(path: &Path, bad: &mut bool) {
    check_entries(&path, bad);
    let (ui, ui_fulldeps) = (path.join("ui"), path.join("ui-fulldeps"));
    let paths = [ui.as_path(), ui_fulldeps.as_path()];
    crate::walk::walk_no_read(&paths, |_, _| false, &mut |entry| {
        let file_path = entry.path();
        if let Some(ext) = file_path.extension().and_then(OsStr::to_str) {
            // files that are neither an expected extension or an exception should not exist
            // they're probably typos or not meant to exist
            if !(EXPECTED_TEST_FILE_EXTENSIONS.contains(&ext)
                || EXTENSION_EXCEPTION_PATHS.iter().any(|path| file_path.ends_with(path)))
            {
                tidy_error!(bad, "file {} has unexpected extension {}", file_path.display(), ext);
            }
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
