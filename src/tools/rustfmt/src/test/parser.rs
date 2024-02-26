use std::io;
use std::path::PathBuf;

use super::read_config;

use crate::modules::{ModuleResolutionError, ModuleResolutionErrorKind};
use crate::{ErrorKind, Input, Session};

#[test]
fn parser_errors_in_submods_are_surfaced() {
    // See also https://github.com/rust-lang/rustfmt/issues/4126
    let filename = "tests/parser/issue-4126/lib.rs";
    let input_file = PathBuf::from(filename);
    let exp_mod_name = "invalid";
    let config = read_config(&input_file);
    let mut session = Session::<io::Stdout>::new(config, None);
    if let Err(ErrorKind::ModuleResolutionError(ModuleResolutionError { module, kind })) =
        session.format(Input::File(filename.into()))
    {
        assert_eq!(&module, exp_mod_name);
        if let ModuleResolutionErrorKind::ParseError {
            file: unparseable_file,
        } = kind
        {
            assert_eq!(
                unparseable_file,
                PathBuf::from("tests/parser/issue-4126/invalid.rs"),
            );
        } else {
            panic!("Expected parser error");
        }
    } else {
        panic!("Expected ModuleResolution operation error");
    }
}

fn assert_parser_error(filename: &str) {
    let file = PathBuf::from(filename);
    let config = read_config(&file);
    let mut session = Session::<io::Stdout>::new(config, None);
    let _ = session.format(Input::File(filename.into())).unwrap();
    assert!(session.has_parsing_errors());
}

#[test]
fn parser_creation_errors_on_entry_new_parser_from_file_panic() {
    // See also https://github.com/rust-lang/rustfmt/issues/4418
    let filename = "tests/parser/issue_4418.rs";
    assert_parser_error(filename);
}

#[test]
fn crate_parsing_errors_on_unclosed_delims() {
    // See also https://github.com/rust-lang/rustfmt/issues/4466
    let filename = "tests/parser/unclosed-delims/issue_4466.rs";
    assert_parser_error(filename);
}

#[test]
fn crate_parsing_stashed_diag() {
    // See also https://github.com/rust-lang/rust/issues/121450
    let filename = "tests/parser/stashed-diag.rs";
    assert_parser_error(filename);
}

#[test]
fn crate_parsing_stashed_diag2() {
    // See also https://github.com/rust-lang/rust/issues/121517
    let filename = "tests/parser/stashed-diag2.rs";
    assert_parser_error(filename);
}
