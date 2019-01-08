extern crate ra_syntax;
extern crate test_utils;
extern crate walkdir;

use std::{
    fmt::Write,
    path::{PathBuf, Component},
};

use test_utils::{project_dir, dir_tests, read_text, collect_tests};
use ra_syntax::{
    SourceFile, AstNode,
    utils::{check_fuzz_invariants, dump_tree},
};

#[test]
fn lexer_tests() {
    dir_tests(&test_data_dir(), &["lexer"], |text, _| {
        let tokens = ra_syntax::tokenize(text);
        dump_tokens(&tokens, text)
    })
}

#[test]
fn parser_tests() {
    dir_tests(
        &test_data_dir(),
        &["parser/inline/ok", "parser/ok"],
        |text, path| {
            let file = SourceFile::parse(text);
            let errors = file.errors();
            assert_eq!(
                &*errors,
                &[] as &[ra_syntax::SyntaxError],
                "There should be no errors in the file {:?}",
                path.display()
            );
            dump_tree(file.syntax())
        },
    );
    dir_tests(
        &test_data_dir(),
        &["parser/err", "parser/inline/err"],
        |text, path| {
            let file = SourceFile::parse(text);
            let errors = file.errors();
            assert_ne!(
                &*errors,
                &[] as &[ra_syntax::SyntaxError],
                "There should be errors in the file {:?}",
                path.display()
            );
            dump_tree(file.syntax())
        },
    );
}

#[test]
fn parser_fuzz_tests() {
    for (_, text) in collect_tests(&test_data_dir(), &["parser/fuzz-failures"]) {
        check_fuzz_invariants(&text)
    }
}

/// Test that Rust-analyzer can parse and validate the rust-analyser
/// TODO: Use this as a benchmark
#[test]
fn self_hosting_parsing() {
    use std::ffi::OsStr;
    let dir = project_dir().join("crates");
    let mut count = 0;
    for entry in walkdir::WalkDir::new(dir)
        .into_iter()
        .filter_entry(|entry| {
            !entry.path().components().any(|component| {
                // Get all files which are not in the crates/ra_syntax/tests/data folder
                component == Component::Normal(OsStr::new("data"))
            })
        })
        .map(|e| e.unwrap())
        .filter(|entry| {
            // Get all `.rs ` files
            !entry.path().is_dir() && (entry.path().extension() == Some(OsStr::new("rs")))
        })
    {
        count += 1;
        let text = read_text(entry.path());
        let node = SourceFile::parse(&text);
        let errors = node.errors();
        assert_eq!(
            &*errors,
            &[],
            "There should be no errors in the file {:?}",
            entry
        );
    }
    assert!(
        count > 30,
        "self_hosting_parsing found too few files - is it running in the right directory?"
    )
}

fn test_data_dir() -> PathBuf {
    project_dir().join("crates/ra_syntax/tests/data")
}

fn dump_tokens(tokens: &[ra_syntax::Token], text: &str) -> String {
    let mut acc = String::new();
    let mut offset = 0;
    for token in tokens {
        let len: u32 = token.len.into();
        let len = len as usize;
        let token_text = &text[offset..offset + len];
        offset += len;
        write!(acc, "{:?} {} {:?}\n", token.kind, token.len, token_text).unwrap()
    }
    acc
}
