extern crate ra_syntax;
#[macro_use]
extern crate test_utils;
extern crate walkdir;

use std::{
    fmt::Write,
    fs,
    path::{Path, PathBuf, Component},
};

use ra_syntax::{
    utils::{check_fuzz_invariants, dump_tree},
    SourceFileNode,
};

#[test]
fn lexer_tests() {
    dir_tests(&["lexer"], |text, _| {
        let tokens = ra_syntax::tokenize(text);
        dump_tokens(&tokens, text)
    })
}

#[test]
fn parser_tests() {
    dir_tests(&["parser/inline/ok", "parser/ok"], |text, path| {
        let file = SourceFileNode::parse(text);
        let errors = file.errors();
        assert_eq!(
            &*errors,
            &[] as &[ra_syntax::SyntaxError],
            "There should be no errors in the file {:?}",
            path.display()
        );
        dump_tree(file.syntax())
    });
    dir_tests(&["parser/err", "parser/inline/err"], |text, path| {
        let file = SourceFileNode::parse(text);
        let errors = file.errors();
        assert_ne!(
            &*errors,
            &[] as &[ra_syntax::SyntaxError],
            "There should be errors in the file {:?}",
            path.display()
        );
        dump_tree(file.syntax())
    });
}

#[test]
fn parser_fuzz_tests() {
    for (_, text) in collect_tests(&["parser/fuzz-failures"]) {
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
        let node = SourceFileNode::parse(&text);
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
        .expect(&format!("File at {:?} should be valid", path))
        .replace("\r\n", "\n")
}

fn dir_tests<F>(paths: &[&str], f: F)
where
    F: Fn(&str, &Path) -> String,
{
    for (path, input_code) in collect_tests(paths) {
        let parse_tree = f(&input_code, &path);
        let path = path.with_extension("txt");
        if !path.exists() {
            println!("\nfile: {}", path.display());
            println!("No .txt file with expected result, creating...\n");
            println!("{}\n{}", input_code, parse_tree);
            fs::write(&path, &parse_tree).unwrap();
            panic!("No expected result")
        }
        let expected = read_text(&path);
        let expected = expected.as_str();
        let parse_tree = parse_tree.as_str();
        assert_equal_text(expected, parse_tree, &path);
    }
}

const REWRITE: bool = false;

fn assert_equal_text(expected: &str, actual: &str, path: &Path) {
    if expected == actual {
        return;
    }
    let dir = project_dir();
    let pretty_path = path.strip_prefix(&dir).unwrap_or_else(|_| path);
    if expected.trim() == actual.trim() {
        println!("whitespace difference, rewriting");
        println!("file: {}\n", pretty_path.display());
        fs::write(path, actual).unwrap();
        return;
    }
    if REWRITE {
        println!("rewriting {}", pretty_path.display());
        fs::write(path, actual).unwrap();
        return;
    }
    assert_eq_text!(expected, actual, "file: {}", pretty_path.display());
}

fn collect_tests(paths: &[&str]) -> Vec<(PathBuf, String)> {
    paths
        .iter()
        .flat_map(|path| {
            let path = test_data_dir().join(path);
            test_from_dir(&path).into_iter()
        })
        .map(|path| {
            let text = read_text(&path);
            (path, text)
        })
        .collect()
}

fn test_from_dir(dir: &Path) -> Vec<PathBuf> {
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

fn project_dir() -> PathBuf {
    let dir = env!("CARGO_MANIFEST_DIR");
    PathBuf::from(dir)
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_owned()
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
