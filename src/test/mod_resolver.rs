use std::io;
use std::path::PathBuf;

use super::read_config;

use crate::{FileName, Input, Session};

#[test]
fn nested_out_of_line_mods_loaded() {
    // See also https://github.com/rust-lang/rustfmt/issues/4874
    let filename = "tests/mod-resolver/issue-4874/main.rs";
    let input_file = PathBuf::from(filename);
    let config = read_config(&input_file);
    let mut session = Session::<io::Stdout>::new(config, None);
    let report = session
        .format(Input::File(filename.into()))
        .expect("Should not have had any execution errors");
    let errors_by_file = &report.internal.borrow().0;
    assert!(errors_by_file.contains_key(&FileName::Real(PathBuf::from(
        "tests/mod-resolver/issue-4874/bar/baz.rs",
    ))));
    assert!(errors_by_file.contains_key(&FileName::Real(PathBuf::from(
        "tests/mod-resolver/issue-4874/foo/qux.rs",
    ))));
}
