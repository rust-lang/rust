use std::io;
use std::path::PathBuf;

use super::read_config;

use crate::{FileName, Input, Session};

fn verify_mod_resolution(input_file_name: &str, exp_misformatted_files: &[&str]) {
    let input_file = PathBuf::from(input_file_name);
    let config = read_config(&input_file);
    let mut session = Session::<io::Stdout>::new(config, None);
    let report = session
        .format(Input::File(input_file_name.into()))
        .expect("Should not have had any execution errors");
    let errors_by_file = &report.internal.borrow().0;
    for exp_file in exp_misformatted_files {
        assert!(errors_by_file.contains_key(&FileName::Real(PathBuf::from(exp_file))));
    }
}

#[test]
fn nested_out_of_line_mods_loaded() {
    // See also https://github.com/rust-lang/rustfmt/issues/4874
    verify_mod_resolution(
        "tests/mod-resolver/issue-4874/main.rs",
        &[
            "tests/mod-resolver/issue-4874/bar/baz.rs",
            "tests/mod-resolver/issue-4874/foo/qux.rs",
        ],
    );
}

#[test]
fn out_of_line_nested_inline_within_out_of_line() {
    // See also https://github.com/rust-lang/rustfmt/issues/5063
    verify_mod_resolution(
        "tests/mod-resolver/issue-5063/main.rs",
        &[
            "tests/mod-resolver/issue-5063/foo/bar/baz.rs",
            "tests/mod-resolver/issue-5063/foo.rs",
        ],
    );
}

#[test]
fn skip_out_of_line_nested_inline_within_out_of_line() {
    // See also https://github.com/rust-lang/rustfmt/issues/5065
    verify_mod_resolution(
        "tests/mod-resolver/skip-files-issue-5065/main.rs",
        &["tests/mod-resolver/skip-files-issue-5065/one.rs"],
    );
}

#[test]
fn fmt_out_of_line_test_modules() {
    // See also https://github.com/rust-lang/rustfmt/issues/5119
    verify_mod_resolution(
        "tests/mod-resolver/test-submodule-issue-5119/tests/test1.rs",
        &[
            "tests/mod-resolver/test-submodule-issue-5119/tests/test1.rs",
            "tests/mod-resolver/test-submodule-issue-5119/tests/test1/sub1.rs",
            "tests/mod-resolver/test-submodule-issue-5119/tests/test1/sub2.rs",
            "tests/mod-resolver/test-submodule-issue-5119/tests/test1/sub3/sub4.rs",
        ],
    )
}

#[test]
fn fallback_and_try_to_resolve_external_submod_relative_to_current_dir_path() {
    // See also https://github.com/rust-lang/rustfmt/issues/5198
    verify_mod_resolution(
        "tests/mod-resolver/issue-5198/lib.rs",
        &[
            "tests/mod-resolver/issue-5198/a.rs",
            "tests/mod-resolver/issue-5198/lib/b.rs",
            "tests/mod-resolver/issue-5198/lib/c/mod.rs",
            "tests/mod-resolver/issue-5198/lib/c/e.rs",
            "tests/mod-resolver/issue-5198/lib/c/d/f.rs",
            "tests/mod-resolver/issue-5198/lib/c/d/g/mod.rs",
        ],
    )
}
