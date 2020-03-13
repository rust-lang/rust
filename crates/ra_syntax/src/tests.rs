use std::{
    fmt::Write,
    path::{Component, Path, PathBuf},
};

use test_utils::{collect_tests, dir_tests, project_dir, read_text};

use crate::{fuzz, tokenize, SourceFile, SyntaxError, TextRange, TextUnit, Token};

#[test]
fn lexer_tests() {
    // FIXME:
    // * Add tests for unicode escapes in byte-character and [raw]-byte-string literals
    // * Add tests for unescape errors

    dir_tests(&test_data_dir(), &["lexer/ok"], |text, path| {
        let (tokens, errors) = tokenize(text);
        assert_errors_are_absent(&errors, path);
        dump_tokens_and_errors(&tokens, &errors, text)
    });
    dir_tests(&test_data_dir(), &["lexer/err"], |text, path| {
        let (tokens, errors) = tokenize(text);
        assert_errors_are_present(&errors, path);
        dump_tokens_and_errors(&tokens, &errors, text)
    });
}

#[test]
fn parse_smoke_test() {
    let code = r##"
fn main() {
    println!("Hello, world!")
}
    "##;

    let parse = SourceFile::parse(code);
    assert!(parse.ok().is_ok());
}

#[test]
fn parser_tests() {
    dir_tests(&test_data_dir(), &["parser/inline/ok", "parser/ok"], |text, path| {
        let parse = SourceFile::parse(text);
        let errors = parse.errors();
        assert_errors_are_absent(&errors, path);
        parse.debug_dump()
    });
    dir_tests(&test_data_dir(), &["parser/err", "parser/inline/err"], |text, path| {
        let parse = SourceFile::parse(text);
        let errors = parse.errors();
        assert_errors_are_present(&errors, path);
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
                // Get all files which are not in the crates/ra_syntax/test_data folder
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
        if let Err(errors) = SourceFile::parse(&text).ok() {
            panic!("Parsing errors:\n{:?}\n{}\n", errors, entry.path().display());
        }
    }
    assert!(
        count > 30,
        "self_hosting_parsing found too few files - is it running in the right directory?"
    )
}

fn test_data_dir() -> PathBuf {
    project_dir().join("crates/ra_syntax/test_data")
}

fn assert_errors_are_present(errors: &[SyntaxError], path: &Path) {
    assert!(!errors.is_empty(), "There should be errors in the file {:?}", path.display());
}
fn assert_errors_are_absent(errors: &[SyntaxError], path: &Path) {
    assert_eq!(
        errors,
        &[] as &[SyntaxError],
        "There should be no errors in the file {:?}",
        path.display(),
    );
}

fn dump_tokens_and_errors(tokens: &[Token], errors: &[SyntaxError], text: &str) -> String {
    let mut acc = String::new();
    let mut offset = TextUnit::from_usize(0);
    for token in tokens {
        let token_len = token.len;
        let token_text = &text[TextRange::offset_len(offset, token.len)];
        offset += token.len;
        writeln!(acc, "{:?} {} {:?}", token.kind, token_len, token_text).unwrap();
    }
    for err in errors {
        writeln!(acc, "> error{:?} token({:?}) msg({})", err.range(), &text[err.range()], err)
            .unwrap();
    }
    acc
}
