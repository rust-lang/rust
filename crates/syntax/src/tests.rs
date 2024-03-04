#[cfg(not(feature = "in-rust-tree"))]
mod ast_src;
#[cfg(not(feature = "in-rust-tree"))]
mod sourcegen_ast;

use std::{
    fs,
    path::{Path, PathBuf},
};

use ast::HasName;
use expect_test::expect_file;
use rayon::prelude::*;
use stdx::format_to_acc;
use test_utils::{bench, bench_fixture, project_root};

use crate::{ast, fuzz, AstNode, SourceFile, SyntaxError};

#[test]
fn parse_smoke_test() {
    let code = r#"
fn main() {
    println!("Hello, world!")
}
    "#;

    let parse = SourceFile::parse(code);
    // eprintln!("{:#?}", parse.syntax_node());
    assert!(parse.ok().is_ok());
}

#[test]
fn benchmark_parser() {
    if std::env::var("RUN_SLOW_BENCHES").is_err() {
        return;
    }

    let data = bench_fixture::glorious_old_parser();
    let tree = {
        let _b = bench("parsing");
        let p = SourceFile::parse(&data);
        assert!(p.errors().is_empty());
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
fn validation_tests() {
    dir_tests(&test_data_dir(), &["parser/validation"], "rast", |text, path| {
        let parse = SourceFile::parse(text);
        let errors = parse.errors();
        assert_errors_are_present(&errors, path);
        parse.debug_dump()
    });
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
    let crates_dir = project_root().join("crates");

    let mut files = ::sourcegen::list_rust_files(&crates_dir);
    files.retain(|path| {
        // Get all files which are not in the crates/syntax/test_data folder
        !path.components().any(|component| component.as_os_str() == "test_data")
    });

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
        let errors = errors.into_iter().fold(String::new(), |mut acc, (path, err)| {
            format_to_acc!(acc, "{}: {:?}\n", path.display(), err[0])
        });
        panic!("Parsing errors:\n{errors}\n");
    }
}

fn test_data_dir() -> PathBuf {
    project_root().join("crates/syntax/test_data")
}

fn assert_errors_are_present(errors: &[SyntaxError], path: &Path) {
    assert!(!errors.is_empty(), "There should be errors in the file {:?}", path.display());
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
    for file in fs::read_dir(dir).unwrap() {
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
        .unwrap_or_else(|_| panic!("File at {path:?} should be valid"))
        .replace("\r\n", "\n")
}
