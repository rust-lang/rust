//! Tidy check to ensure below in UI test directories:
//! - there are no stray `.stderr` files

use std::collections::BTreeSet;
use std::ffi::OsStr;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

const ISSUES_TXT_HEADER: &str = r#"============================================================
    ⚠️⚠️⚠️NOTHING SHOULD EVER BE ADDED TO THIS LIST⚠️⚠️⚠️
============================================================
"#;

pub fn check(root_path: &Path, bless: bool, bad: &mut bool) {
    let path = &root_path.join("tests");

    // the list of files in ui tests that are allowed to start with `issue-XXXX`
    // BTreeSet because we would like a stable ordering so --bless works
    let mut prev_line = "";
    let mut is_sorted = true;
    let allowed_issue_names: BTreeSet<_> = include_str!("issues.txt")
        .strip_prefix(ISSUES_TXT_HEADER)
        .unwrap()
        .lines()
        .inspect(|&line| {
            if prev_line > line {
                is_sorted = false;
            }

            prev_line = line;
        })
        .collect();

    if !is_sorted && !bless {
        tidy_error!(
            bad,
            "`src/tools/tidy/src/issues.txt` is not in order, mostly because you modified it manually,
            please only update it with command `x test tidy --bless`"
        );
    }

    deny_new_top_level_ui_tests(bad, &path.join("ui"));

    let remaining_issue_names = recursively_check_ui_tests(bad, path, &allowed_issue_names);

    // if there are any file names remaining, they were moved on the fs.
    // our data must remain up to date, so it must be removed from issues.txt
    // do this automatically on bless, otherwise issue a tidy error
    if bless && (!remaining_issue_names.is_empty() || !is_sorted) {
        let tidy_src = root_path.join("src/tools/tidy/src");
        // instead of overwriting the file, recreate it and use an "atomic rename"
        // so we don't bork things on panic or a contributor using Ctrl+C
        let blessed_issues_path = tidy_src.join("issues_blessed.txt");
        let mut blessed_issues_txt = fs::File::create(&blessed_issues_path).unwrap();
        blessed_issues_txt.write_all(ISSUES_TXT_HEADER.as_bytes()).unwrap();
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

fn deny_new_top_level_ui_tests(bad: &mut bool, tests_path: &Path) {
    // See <https://github.com/rust-lang/compiler-team/issues/902> where we propose banning adding
    // new ui tests *directly* under `tests/ui/`. For more context, see:
    //
    // - <https://github.com/rust-lang/rust/issues/73494>
    // - <https://github.com/rust-lang/rust/issues/133895>

    let top_level_ui_tests = walkdir::WalkDir::new(tests_path)
        .min_depth(1)
        .max_depth(1)
        .follow_links(false)
        .same_file_system(true)
        .into_iter()
        .flatten()
        .filter(|e| {
            let file_name = e.file_name();
            file_name != ".gitattributes" && file_name != "README.md"
        })
        .filter(|e| !e.file_type().is_dir());
    for entry in top_level_ui_tests {
        tidy_error!(
            bad,
            "ui tests should be added under meaningful subdirectories: `{}`",
            entry.path().display()
        )
    }
}

fn recursively_check_ui_tests<'issues>(
    bad: &mut bool,
    path: &Path,
    allowed_issue_names: &'issues BTreeSet<&'issues str>,
) -> BTreeSet<&'issues str> {
    let mut remaining_issue_names: BTreeSet<&str> = allowed_issue_names.clone();

    let (ui, ui_fulldeps) = (path.join("ui"), path.join("ui-fulldeps"));
    let paths = [ui.as_path(), ui_fulldeps.as_path()];
    crate::walk::walk_no_read(&paths, |_, _| false, &mut |entry| {
        let file_path = entry.path();
        if let Some(ext) = file_path.extension().and_then(OsStr::to_str) {
            check_unexpected_extension(bad, file_path, ext);

            // NB: We do not use file_stem() as some file names have multiple `.`s and we
            // must strip all of them.
            let testname =
                file_path.file_name().unwrap().to_str().unwrap().split_once('.').unwrap().0;
            if ext == "stderr" || ext == "stdout" || ext == "fixed" {
                check_stray_output_snapshot(bad, file_path, testname);
                check_empty_output_snapshot(bad, file_path);
            }

            deny_new_nondescriptive_test_names(
                bad,
                path,
                &mut remaining_issue_names,
                file_path,
                testname,
                ext,
            );
        }
    });
    remaining_issue_names
}

fn check_unexpected_extension(bad: &mut bool, file_path: &Path, ext: &str) {
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

    // files that are neither an expected extension or an exception should not exist
    // they're probably typos or not meant to exist
    if !(EXPECTED_TEST_FILE_EXTENSIONS.contains(&ext)
        || EXTENSION_EXCEPTION_PATHS.iter().any(|path| file_path.ends_with(path)))
    {
        tidy_error!(bad, "file {} has unexpected extension {}", file_path.display(), ext);
    }
}

fn check_stray_output_snapshot(bad: &mut bool, file_path: &Path, testname: &str) {
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
}

fn check_empty_output_snapshot(bad: &mut bool, file_path: &Path) {
    if let Ok(metadata) = fs::metadata(file_path)
        && metadata.len() == 0
    {
        tidy_error!(bad, "Empty file with UI testing output: {:?}", file_path);
    }
}

fn deny_new_nondescriptive_test_names(
    bad: &mut bool,
    path: &Path,
    remaining_issue_names: &mut BTreeSet<&str>,
    file_path: &Path,
    testname: &str,
    ext: &str,
) {
    if ext == "rs"
        && let Some(test_name) = static_regex!(r"^issues?[-_]?(\d{3,})").captures(testname)
    {
        // these paths are always relative to the passed `path` and always UTF8
        let stripped_path = file_path
            .strip_prefix(path)
            .unwrap()
            .to_str()
            .unwrap()
            .replace(std::path::MAIN_SEPARATOR_STR, "/");

        if !remaining_issue_names.remove(stripped_path.as_str())
            && !stripped_path.starts_with("ui/issues/")
        {
            tidy_error!(
                bad,
                "file `tests/{stripped_path}` must begin with a descriptive name, consider `{{reason}}-issue-{issue_n}.rs`",
                issue_n = &test_name[1],
            );
        }
    }
}
