//! Tidy check to ensure below in UI test directories:
//! - the number of entries in each directory must be less than `ENTRY_LIMIT`
//! - there are no stray `.stderr` files

use crate::ui_tests::ui_test_headers::{HeaderAction, HeaderError, LineAction};
use ignore::Walk;
use std::collections::HashMap;
use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::{fs, io};

mod ui_test_headers;

const ENTRY_LIMIT: usize = 900;
// FIXME: The following limits should be reduced eventually.
const ISSUES_ENTRY_LIMIT: usize = 1893;
const ROOT_ENTRY_LIMIT: usize = 872;

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
        // There are files in these directories that do not have extensions for a variety of reasons. Ignore them.
        let Some(ext) = file_path.extension().and_then(OsStr::to_str) else {
            return;
        };

        // files that are neither an expected extension or an exception should not exist
        // they're probably typos or not meant to exist
        if !(EXPECTED_TEST_FILE_EXTENSIONS.contains(&ext)
            || EXTENSION_EXCEPTION_PATHS.iter().any(|path| file_path.ends_with(path)))
        {
            tidy_error!(bad, "file {} has unexpected extension {}", file_path.display(), ext);
        }

        // NB: We do not use file_stem() as some file names have multiple `.`s and we
        // must strip all of them.
        let testname = file_path.file_name().unwrap().to_str().unwrap().split_once('.').unwrap().0;
        match ext {
            "stderr" | "stdout" => {
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
                if !file_path.with_file_name(testname).with_extension("rs").exists()
                    && !testname.contains("ignore-tidy")
                {
                    tidy_error!(bad, "Stray file with UI testing output: {:?}", file_path);
                }

                if let Ok(metadata) = fs::metadata(file_path) {
                    if metadata.len() == 0 {
                        tidy_error!(bad, "Empty file with UI testing output: {:?}", file_path);
                    }
                }
            }
            "rs" => {
                // FIXME (ui_test): make this configurable somehow
                let mode = HeaderCheckMode::Error;
                // let mode = HeaderCheckMode::Fix;
                check_ui_test_headers(bad, file_path, mode);
            }
            _ => {}
        }
    });
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HeaderCheckMode {
    /// Emit a tidy error if a header is incorrect.
    Error,
    /// Correct the file and do not emit a tidy error.
    Fix,
}

/// Checks that a test file uses the new ui_test style headers where possible.
/// Can be configured to either emit an error or fix the issues.
fn check_ui_test_headers(bad: &mut bool, file_path: &Path, mode: HeaderCheckMode) {
    match ui_test_headers::check_file_headers(file_path) {
        Ok(()) => {}
        Err(HeaderError::IoError(io_err)) => {
            tidy_error!(
                bad,
                "An IO error occurred when processing headers for {}: {}",
                file_path.display(),
                io_err
            );
        }
        Err(HeaderError::InvalidHeader { bad_lines }) => {
            match mode {
                HeaderCheckMode::Error => {
                    // Sanity check, this should not be possible.
                    if bad_lines.len() == 0 {
                        unreachable!("");
                    }

                    *bad = emit_header_errors(file_path, bad_lines);
                }
                HeaderCheckMode::Fix => {
                    if let Err(e) = fix_header_errors(file_path, bad_lines) {
                        tidy_error!(
                            bad,
                            "An IO error occurred while fixing headers for {}: {}",
                            file_path.display(),
                            e
                        );
                    }
                }
            }
        }
    }
}

/// Emits errors for the header lines specified. Returns whether any errors were emitted
fn emit_header_errors(file_path: &Path, bad_lines: Vec<HeaderAction>) -> bool {
    let mut bad = false;
    for action in bad_lines {
        let err = action.error_message();
        tidy_error!(
            &mut bad,
            "invalid test header\n    {}:{}\n    {}",
            file_path.display(),
            action.line_num(),
            err
        );
    }
    bad
}

fn fix_header_errors(file_path: &Path, bad_lines: Vec<HeaderAction>) -> io::Result<()> {
    // Process each header error into a replacement for a line.
    let line_replacements = bad_lines
        .into_iter()
        .map(|header_action| {
            (
                header_action.line_num(),
                match header_action.action() {
                    LineAction::UseUiTestComment => {
                        replace_compiletest_comment(header_action.line()).unwrap()
                    }
                    LineAction::MigrateToUiTest { compiletest_name, ui_test_name } => {
                        // Replace comment type first, then the name range specified.
                        let mut new_line =
                            replace_compiletest_comment(header_action.line()).unwrap();
                        // This is always a directive that contains the compiletest name.
                        let name_start = new_line.find(compiletest_name.as_str()).unwrap();
                        new_line.replace_range(
                            name_start..(name_start + compiletest_name.len()),
                            ui_test_name.as_str(),
                        );
                        new_line
                    }
                    LineAction::UseUITestName { compiletest_name, ui_test_name } => {
                        // This is always a directive that contains the compiletest name.
                        let name_start =
                            header_action.line().find(compiletest_name.as_str()).unwrap();
                        let mut new_line = header_action.line().to_string();
                        new_line.replace_range(
                            name_start..(name_start + compiletest_name.len()),
                            ui_test_name.as_str(),
                        );
                        new_line
                    }
                    LineAction::Error { message } => todo!(),
                },
            )
        })
        .collect::<HashMap<_, _>>();

    let file_contents = fs::read_to_string(file_path)?;

    // Replace each line in the contents of the file that there is an entry for.
    let replaced_contents = file_contents
        // split_inclusive here because we want each line to still have its newline to be
        // joined. The line replacements also keep their newline.
        .split_inclusive('\n')
        .enumerate()
        .map(|(line_num_zero_idx, line)| {
            // enumerate is 0-indexed, but the entries for line numbers are 1-indexed.
            line_replacements.get(&(line_num_zero_idx + 1)).map(|s| s.as_str()).unwrap_or(line)
        })
        .collect::<String>();
    // dbg!(&replaced_contents);

    println!("Writing fixed file {}", file_path.display());

    // Return whether the file was successfully written.
    fs::write(file_path, replaced_contents)
}

/// Replace the comment portion of a compiletest style header with a ui_test style comment.
/// Returns None if the comment did not start with a compiletest style comment.
fn replace_compiletest_comment(line: &str) -> Option<String> {
    // Find the first character that's not part of the comment start, and replace
    let end_pos = line
        .char_indices()
        // This match is more permissive than the start of the comments should be,
        // but since this is in iter_header, all of these lines are well formed comments.
        .skip_while(|&(_, c)| matches!(c, '/' | ' '))
        .next()?
        .0;

    let mut new_line = line.to_string();
    // Replace range is exclusive because the found end pos is the first non-start character
    new_line.replace_range(0..end_pos, "//@");

    Some(new_line)
}
