//! Tidy check to ensure below in UI test directories:
//! - the number of entries in each directory must be less than `ENTRY_LIMIT`
//! - there are no stray `.stderr` files

use std::collections::{BTreeSet, HashMap};
use std::ffi::OsStr;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use ignore::Walk;

// FIXME: GitHub's UI truncates file lists that exceed 1000 entries, so these
// should all be 1000 or lower. Limits significantly smaller than 1000 are also
// desirable, because large numbers of files are unwieldy in general. See issue
// #73494.
const ENTRY_LIMIT: u32 = 901;
// FIXME: The following limits should be reduced eventually.

const ISSUES_ENTRY_LIMIT: u32 = 1619;

const EXPECTED_TEST_FILE_EXTENSIONS: &[&str] = &[
    "rs",     // test source files
    "stderr", // expected stderr file, corresponds to a rs file
    "svg",    // expected svg file, corresponds to a rs file, equivalent to stderr
    "stdout", // expected stdout file, corresponds to a rs file
    "fixed",  // expected source file after applying fixes
    "md",     // test directory descriptions
    "ftl",    // translation tests
];

const EXTENSION_EXCEPTION_PATHS: &[&str] = &[
    "tests/ui/asm/named-asm-labels.s", // loading an external asm file to test named labels lint
    "tests/ui/codegen/mismatched-data-layout.json", // testing mismatched data layout w/ custom targets
    "tests/ui/check-cfg/my-awesome-platform.json",  // testing custom targets with cfgs
    "tests/ui/argfile/commandline-argfile-badutf8.args", // passing args via a file
    "tests/ui/argfile/commandline-argfile.args",    // passing args via a file
    "tests/ui/crate-loading/auxiliary/libfoo.rlib", // testing loading a manually created rlib
    "tests/ui/include-macros/data.bin", // testing including data with the include macros
    "tests/ui/include-macros/file.txt", // testing including data with the include macros
    "tests/ui/macros/macro-expanded-include/file.txt", // testing including data with the include macros
    "tests/ui/macros/not-utf8.bin", // testing including data with the include macros
    "tests/ui/macros/syntax-extension-source-utils-files/includeme.fragment", // more include
    "tests/ui/proc-macro/auxiliary/included-file.txt", // more include
    "tests/ui/unpretty/auxiliary/data.txt", // more include
    "tests/ui/invalid/foo.natvis.xml", // sample debugger visualizer
    "tests/ui/sanitizer/dataflow-abilist.txt", // dataflow sanitizer ABI list file
    "tests/ui/shell-argfiles/shell-argfiles.args", // passing args via a file
    "tests/ui/shell-argfiles/shell-argfiles-badquotes.args", // passing args via a file
    "tests/ui/shell-argfiles/shell-argfiles-via-argfile-shell.args", // passing args via a file
    "tests/ui/shell-argfiles/shell-argfiles-via-argfile.args", // passing args via a file
    "tests/ui/std/windows-bat-args1.bat", // tests escaping arguments through batch files
    "tests/ui/std/windows-bat-args2.bat", // tests escaping arguments through batch files
    "tests/ui/std/windows-bat-args3.bat", // tests escaping arguments through batch files
];

fn check_entries(tests_path: &Path, bad: &mut bool) {
    let mut directories: HashMap<PathBuf, u32> = HashMap::new();

    for dir in Walk::new(&tests_path.join("ui")) {
        if let Ok(entry) = dir {
            let parent = entry.path().parent().unwrap().to_path_buf();
            *directories.entry(parent).or_default() += 1;
        }
    }

    let (mut max, mut max_issues) = (0, 0);
    for (dir_path, count) in directories {
        let is_issues_dir = tests_path.join("ui/issues") == dir_path;
        let (limit, maxcnt) = if is_issues_dir {
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
    if ISSUES_ENTRY_LIMIT > max_issues {
        tidy_error!(
            bad,
            "`ISSUES_ENTRY_LIMIT` is too high (is {ISSUES_ENTRY_LIMIT}, should be {max_issues})"
        );
    }
}

pub fn check(root_path: &Path, bless: bool, bad: &mut bool) {
    let issues_txt_header = r#"============================================================
    ⚠️⚠️⚠️NOTHING SHOULD EVER BE ADDED TO THIS LIST⚠️⚠️⚠️
============================================================
"#;

    let path = &root_path.join("tests");
    check_entries(&path, bad);

    // the list of files in ui tests that are allowed to start with `issue-XXXX`
    // BTreeSet because we would like a stable ordering so --bless works
    let mut prev_line = "";
    let mut is_sorted = true;
    let allowed_issue_names: BTreeSet<_> = include_str!("issues.txt")
        .strip_prefix(issues_txt_header)
        .unwrap()
        .lines()
        .map(|line| {
            if prev_line > line {
                is_sorted = false;
            }

            prev_line = line;
            line
        })
        .collect();

    if !is_sorted && !bless {
        tidy_error!(
            bad,
            "`src/tools/tidy/src/issues.txt` is not in order, mostly because you modified it manually,
            please only update it with command `x test tidy --bless`"
        );
    }

    let mut remaining_issue_names: BTreeSet<&str> = allowed_issue_names.clone();

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

            // NB: We do not use file_stem() as some file names have multiple `.`s and we
            // must strip all of them.
            let testname =
                file_path.file_name().unwrap().to_str().unwrap().split_once('.').unwrap().0;
            if ext == "stderr" || ext == "stdout" || ext == "fixed" {
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

            if ext == "rs" {
                if let Some(test_name) = static_regex!(r"^issues?[-_]?(\d{3,})").captures(testname)
                {
                    // these paths are always relative to the passed `path` and always UTF8
                    let stripped_path = file_path
                        .strip_prefix(path)
                        .unwrap()
                        .to_str()
                        .unwrap()
                        .replace(std::path::MAIN_SEPARATOR_STR, "/");

                    if !remaining_issue_names.remove(stripped_path.as_str()) {
                        tidy_error!(
                            bad,
                            "file `tests/{stripped_path}` must begin with a descriptive name, consider `{{reason}}-issue-{issue_n}.rs`",
                            issue_n = &test_name[1],
                        );
                    }
                }
            }
        }
    });

    // if there are any file names remaining, they were moved on the fs.
    // our data must remain up to date, so it must be removed from issues.txt
    // do this automatically on bless, otherwise issue a tidy error
    if bless && (!remaining_issue_names.is_empty() || !is_sorted) {
        let tidy_src = root_path.join("src/tools/tidy/src");
        // instead of overwriting the file, recreate it and use an "atomic rename"
        // so we don't bork things on panic or a contributor using Ctrl+C
        let blessed_issues_path = tidy_src.join("issues_blessed.txt");
        let mut blessed_issues_txt = fs::File::create(&blessed_issues_path).unwrap();
        blessed_issues_txt.write(issues_txt_header.as_bytes()).unwrap();
        // If we changed paths to use the OS separator, reassert Unix chauvinism for blessing.
        for filename in allowed_issue_names.difference(&remaining_issue_names) {
            writeln!(blessed_issues_txt, "{filename}").unwrap();
        }
        let old_issues_path = tidy_src.join("issues.txt");
        fs::rename(blessed_issues_path, old_issues_path).unwrap();
    } else {
        for file_name in remaining_issue_names {
            let mut p = PathBuf::from(path);
            p.push(file_name);
            tidy_error!(
                bad,
                "file `{}` no longer exists and should be removed from the exclusions in `src/tools/tidy/src/issues.txt`",
                p.display()
            );
        }
    }
}
