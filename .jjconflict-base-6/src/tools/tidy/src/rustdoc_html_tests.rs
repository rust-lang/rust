//! Tidy check to ensure that rustdoc GUI tests start with a small description.

use std::collections::BTreeSet;
use std::path::Path;

use crate::diagnostics::{CheckId, TidyCtx};

pub fn check(path: &Path, tidy_ctx: TidyCtx) {
    let path = path.join("rustdoc-html");
    let mut check = tidy_ctx.start_check(CheckId::new("rustdoc_html_tests").path(&path));

    // The list of subdirectories in rustdoc-html tests.
    // Compare previous subdirectory with current subdirectory
    // to sync with `tests/rustdoc-html/README.md`.
    let mut prev_line = String::new();
    let documented_subdirs: BTreeSet<_> = include_str!("../../../../tests/rustdoc-html/README.md")
        .lines()
        .filter_map(|line| {
            static_regex!(r"^##.*?`(?<dir>[^`]+)`").captures(line).map(|cap| {
                let dir = &cap["dir"];
                // FIXME(reddevilmidzy) normalize subdirs title in tests/rustdoc-html/README.md
                if dir.ends_with('/') {
                    dir.strip_suffix('/').unwrap().to_string()
                } else {
                    dir.to_string()
                }
            })
        })
        .inspect(|line| {
            if prev_line.as_str() > line.as_str() {
                check.error(&format!(
                    "`tests/rustdoc-html/README.md` is not in order: {prev_line:?} should be after {line:?}"
                ));
            }

            prev_line = line.clone();
        })
        .collect();

    let filesystem_subdirs = collect_rustdoc_html_tests_subdirs(&path);
    let is_modified = !filesystem_subdirs.eq(&documented_subdirs);

    if is_modified {
        for directory in documented_subdirs.symmetric_difference(&filesystem_subdirs) {
            if documented_subdirs.contains(directory) {
                check.error(format!(
                   "rustdoc-html subdirectory `{directory}` is listed in `tests/rustdoc-html/README.md` but does not exist in the filesystem"
               ));
            } else {
                check.error(format!(
                   "rustdoc-html subdirectory `{directory}` exists in the filesystem but is not documented in `tests/rustdoc-html/README.md`"
               ));
            }
        }
        check.error(
           "`tests/rustdoc-html/README.md` subdirectory listing is out of sync with the filesystem. \
            Please add or remove subdirectory entries (## headers with backtick-wrapped names) to match the actual directories in `tests/rustdoc-html/`"
       );
    }
}

fn collect_rustdoc_html_tests_subdirs(path: &Path) -> BTreeSet<String> {
    let entries = std::fs::read_dir(path).unwrap();

    entries
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.is_dir() && path.file_name().is_some_and(|name| name != "auxiliary"))
        .map(|dir_path| {
            let dir_path = dir_path.strip_prefix(path).unwrap();
            format!(
                "tests/rustdoc-html/{}",
                dir_path.to_string_lossy().replace(std::path::MAIN_SEPARATOR_STR, "/")
            )
        })
        .collect()
}
