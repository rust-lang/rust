use std::{
    fmt::Write,
    path::{Component, PathBuf},
};

use test_utils::{collect_tests, dir_tests, project_dir, read_text};

use crate::{fuzz, SourceFile};

#[test]
fn lexer_tests() {
    dir_tests(&test_data_dir(), &["lexer"], |text, _| {
        let tokens = crate::tokenize(text);
        dump_tokens(&tokens, text)
    })
}

#[test]
fn parser_tests() {
    dir_tests(&test_data_dir(), &["parser/inline/ok", "parser/ok"], |text, path| {
        let parse = SourceFile::parse(text);
        let errors = parse.errors();
        assert_eq!(
            errors,
            &[] as &[crate::SyntaxError],
            "There should be no errors in the file {:?}",
            path.display(),
        );
        parse.debug_dump()
    });
    dir_tests(&test_data_dir(), &["parser/err", "parser/inline/err"], |text, path| {
        let parse = SourceFile::parse(text);
        let errors = parse.errors();
        assert!(!errors.is_empty(), "There should be errors in the file {:?}", path.display());
        parse.debug_dump()
    });
}

#[test]
fn parser_fuzz_tests() {
    for (_, text) in collect_tests(&test_data_dir(), &["parser/fuzz-failures"]) {
        fuzz::check_parser(&text)
    }
}

#[test]
fn reparse_fuzz_tests() {
    for (_, text) in collect_tests(&test_data_dir(), &["reparse/fuzz-failures"]) {
        let check = fuzz::CheckReparse::from_data(text.as_bytes()).unwrap();
        println!("{:?}", check);
        check.run();
    }
}

/// Test that Rust-analyzer can parse and validate the rust-analyzer
/// FIXME: Use this as a benchmark
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
                component == Component::Normal(OsStr::new("test_data"))
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
        SourceFile::parse(&text).ok().expect("There should be no errors in the file");
    }
    assert!(
        count > 30,
        "self_hosting_parsing found too few files - is it running in the right directory?"
    )
}

fn test_data_dir() -> PathBuf {
    project_dir().join("crates/ra_syntax/test_data")
}

fn dump_tokens(tokens: &[crate::Token], text: &str) -> String {
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
