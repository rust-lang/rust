//! Tidy check to ensure that all `compiler/` crates have `[lints] workspace =
//! true` and therefore inherit the standard lints.

use std::path::Path;

use crate::walk::{filter_dirs, walk};

pub fn check(path: &Path, bad: &mut bool) {
    walk(path, |path, _is_dir| filter_dirs(path), &mut |entry, contents| {
        let file = entry.path();
        let filename = file.file_name().unwrap();
        if filename != "Cargo.toml" {
            return;
        }

        let has_lints_line = contents.lines().any(|line| line.trim() == "[lints]");
        let has_workspace_line = contents.lines().any(|line| line.trim() == "workspace = true");

        if !has_lints_line {
            tidy_error!(bad, "{} doesn't have a `[lints]` line", file.display());
        }
        if !has_lints_line || !has_workspace_line {
            tidy_error!(bad, "{} doesn't have a `workspace = true` line", file.display());
        }
    });
}
