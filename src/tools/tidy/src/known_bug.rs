//! Tidy check to ensure that tests inside 'tests/crashes' have a '@known-bug' directive.

use std::path::Path;

use crate::walk::*;

pub fn check(filepath: &Path, bad: &mut bool) {
    walk(filepath, |path, _is_dir| filter_not_rust(path), &mut |entry, contents| {
        let file: &Path = entry.path();

        // files in "auxiliary" do not need to crash by themselves
        let test_path_segments =
            file.iter().map(|s| s.to_string_lossy().into()).collect::<Vec<String>>();
        let test_path_segments_str =
            test_path_segments.iter().map(|s| s.as_str()).collect::<Vec<&str>>();

        if !matches!(
            test_path_segments_str[..],
            [.., "tests", "crashes", "auxiliary", _aux_file_rs]
        ) && !contents.lines().any(|line| line.starts_with("//@ known-bug: "))
        {
            tidy_error!(
                bad,
                "{} crash/ice test does not have a \"//@ known-bug: \" directive",
                file.display()
            );
        }
    });
}
