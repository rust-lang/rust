use std::{
    fmt::Write,
    fs,
    path::{Path, PathBuf},
};

use ast::NameOwner;
use expect_test::expect_file;
use rayon::prelude::*;
use test_utils::{bench, bench_fixture, project_root, skip_slow_tests};

use crate::{ast, fuzz, tokenize, AstNode, SourceFile, SyntaxError, TextRange, TextSize, Token};

#[test]
fn lexer_tests() {
    // FIXME:
    // * Add tests for unicode escapes in byte-character and [raw]-byte-string literals
    // * Add tests for unescape errors

    dir_tests(&test_data_dir(), &["lexer/ok"], "txt", |text, path| {
        let (tokens, errors) = tokenize(text);
        assert_errors_are_absent(&errors, path);
        dump_tokens_and_errors(&tokens, &errors, text)
    });
    dir_tests(&test_data_dir(), &["lexer/err"], "txt", |text, path| {
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
    // eprintln!("{:#?}", parse.syntax_node());
    assert!(parse.ok().is_ok());
}

#[test]
fn benchmark_parser() {
    if skip_slow_tests() {
        return;
    }
    let data = bench_fixture::glorious_old_parser();
    let tree = {
        let _b = bench("parsing");
        let p = SourceFile::parse(&data);
        assert!(p.errors.is_empty());
        assert_eq!(p.tree().syntax.text_range().len(), 352474.into());
        p.tree()
    };

    {
        let _b = bench("tree traversal");
        let fn_names =
            tree.syntax().descendants().filter_map(ast::Fn::cast).filter_map(|f| f.name()).count();
        assert_eq!(fn_names, 268);
    }
}

#[test]
fn parser_tests() {
    dir_tests(&test_data_dir(), &["parser/inline/ok", "parser/ok"], "rast", |text, path| {
        let parse = SourceFile::parse(text);
        let errors = parse.errors();
        assert_errors_are_absent(&errors, path);
        parse.debug_dump()
    });
    dir_tests(&test_data_dir(), &["parser/err", "parser/inline/err"], "rast", |text, path| {
        let parse = SourceFile::parse(text);
        let errors = parse.errors();
        assert_errors_are_present(&errors, path);
        parse.debug_dump()
    });
}

#[test]
fn expr_parser_tests() {
    fragment_parser_dir_test(
        &["parser/fragments/expr/ok"],
        &["parser/fragments/expr/err"],
        crate::ast::Expr::parse,
    );
}

#[test]
fn path_parser_tests() {
    fragment_parser_dir_test(
        &["parser/fragments/path/ok"],
        &["parser/fragments/path/err"],
        crate::ast::Path::parse,
    );
}

#[test]
fn pattern_parser_tests() {
    fragment_parser_dir_test(
        &["parser/fragments/pattern/ok"],
        &["parser/fragments/pattern/err"],
        crate::ast::Pat::parse,
    );
}

#[test]
fn item_parser_tests() {
    fragment_parser_dir_test(
        &["parser/fragments/item/ok"],
        &["parser/fragments/item/err"],
        crate::ast::Item::parse,
    );
}

#[test]
fn type_parser_tests() {
    fragment_parser_dir_test(
        &["parser/fragments/type/ok"],
        &["parser/fragments/type/err"],
        crate::ast::Type::parse,
    );
}

#[test]
fn stmt_parser_tests() {
    fragment_parser_dir_test(
        &["parser/fragments/stmt/ok"],
        &["parser/fragments/stmt/err"],
        crate::ast::Stmt::parse,
    );
}

#[test]
fn parser_fuzz_tests() {
    for (_, text) in collect_rust_files(&test_data_dir(), &["parser/fuzz-failures"]) {
        fuzz::check_parser(&text)
    }
}

#[test]
fn reparse_fuzz_tests() {
    for (_, text) in collect_rust_files(&test_data_dir(), &["reparse/fuzz-failures"]) {
        let check = fuzz::CheckReparse::from_data(text.as_bytes()).unwrap();
        check.run();
    }
}

/// Test that Rust-analyzer can parse and validate the rust-analyzer
#[test]
fn self_hosting_parsing() {
    let dir = project_root().join("crates");
    let files = walkdir::WalkDir::new(dir)
        .into_iter()
        .filter_entry(|entry| {
            // Get all files which are not in the crates/syntax/test_data folder
            !entry.path().components().any(|component| component.as_os_str() == "test_data")
        })
        .map(|e| e.unwrap())
        .filter(|entry| {
            // Get all `.rs ` files
            !entry.path().is_dir() && (entry.path().extension().unwrap_or_default() == "rs")
        })
        .map(|entry| entry.into_path())
        .collect::<Vec<_>>();
    assert!(
        files.len() > 100,
        "self_hosting_parsing found too few files - is it running in the right directory?"
    );

    let errors = files
        .into_par_iter()
        .filter_map(|file| {
            let text = read_text(&file);
            match SourceFile::parse(&text).ok() {
                Ok(_) => None,
                Err(err) => Some((file, err)),
            }
        })
        .collect::<Vec<_>>();

    if !errors.is_empty() {
        let errors = errors
            .into_iter()
            .map(|(path, err)| format!("{}: {:?}\n", path.display(), err))
            .collect::<String>();
        panic!("Parsing errors:\n{}\n", errors);
    }
}

fn test_data_dir() -> PathBuf {
    project_root().join("crates/syntax/test_data")
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
    let mut offset: TextSize = 0.into();
    for token in tokens {
        let token_len = token.len;
        let token_text = &text[TextRange::at(offset, token.len)];
        offset += token.len;
        writeln!(acc, "{:?} {:?} {:?}", token.kind, token_len, token_text).unwrap();
    }
    for err in errors {
        writeln!(acc, "> error{:?} token({:?}) msg({})", err.range(), &text[err.range()], err)
            .unwrap();
    }
    acc
}

fn fragment_parser_dir_test<T, F>(ok_paths: &[&str], err_paths: &[&str], f: F)
where
    T: crate::AstNode,
    F: Fn(&str) -> Result<T, ()>,
{
    dir_tests(&test_data_dir(), ok_paths, "rast", |text, path| {
        if let Ok(node) = f(text) {
            format!("{:#?}", crate::ast::AstNode::syntax(&node))
        } else {
            panic!("Failed to parse '{:?}'", path);
        }
    });
    dir_tests(&test_data_dir(), err_paths, "rast", |text, path| {
        if f(text).is_ok() {
            panic!("'{:?}' successfully parsed when it should have errored", path);
        } else {
            "ERROR\n".to_owned()
        }
    });
}

/// Calls callback `f` with input code and file paths for each `.rs` file in `test_data_dir`
/// subdirectories defined by `paths`.
///
/// If the content of the matching output file differs from the output of `f()`
/// the test will fail.
///
/// If there is no matching output file it will be created and filled with the
/// output of `f()`, but the test will fail.
fn dir_tests<F>(test_data_dir: &Path, paths: &[&str], outfile_extension: &str, f: F)
where
    F: Fn(&str, &Path) -> String,
{
    for (path, input_code) in collect_rust_files(test_data_dir, paths) {
        let actual = f(&input_code, &path);
        let path = path.with_extension(outfile_extension);
        expect_file![path].assert_eq(&actual)
    }
}

/// Collects all `.rs` files from `dir` subdirectories defined by `paths`.
fn collect_rust_files(root_dir: &Path, paths: &[&str]) -> Vec<(PathBuf, String)> {
    paths
        .iter()
        .flat_map(|path| {
            let path = root_dir.to_owned().join(path);
            rust_files_in_dir(&path).into_iter()
        })
        .map(|path| {
            let text = read_text(&path);
            (path, text)
        })
        .collect()
}

/// Collects paths to all `.rs` files from `dir` in a sorted `Vec<PathBuf>`.
fn rust_files_in_dir(dir: &Path) -> Vec<PathBuf> {
    let mut acc = Vec::new();
    for file in fs::read_dir(&dir).unwrap() {
        let file = file.unwrap();
        let path = file.path();
        if path.extension().unwrap_or_default() == "rs" {
            acc.push(path);
        }
    }
    acc.sort();
    acc
}

/// Read file and normalize newlines.
///
/// `rustc` seems to always normalize `\r\n` newlines to `\n`:
///
/// ```
/// let s = "
/// ";
/// assert_eq!(s.as_bytes(), &[10]);
/// ```
///
/// so this should always be correct.
fn read_text(path: &Path) -> String {
    fs::read_to_string(path)
        .unwrap_or_else(|_| panic!("File at {:?} should be valid", path))
        .replace("\r\n", "\n")
}
