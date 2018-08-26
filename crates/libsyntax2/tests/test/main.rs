extern crate libsyntax2;
#[macro_use]
extern crate test_utils;
extern crate walkdir;

use std::{
    fs,
    path::{Path, PathBuf},
    fmt::Write,
};

use test_utils::extract_range;
use libsyntax2::{
    File, AtomEdit,
    utils::dump_tree,
};

#[test]
fn lexer_tests() {
    dir_tests(&["lexer"], |text| {
        let tokens = libsyntax2::tokenize(text);
        dump_tokens(&tokens, text)
    })
}

#[test]
fn parser_tests() {
    dir_tests(&["parser/inline", "parser/ok", "parser/err"], |text| {
        eprintln!("\n{}\n", text);
        let file = File::parse(text);
        dump_tree(file.syntax())
    })
}

#[test]
fn reparse_test() {
    fn do_check(before: &str, replace_with: &str) {
        let (range, before) = extract_range(before);
        let after = libsyntax2::replace_range(before.clone(), range, replace_with);

        let fully_reparsed = File::parse(&after);
        let incrementally_reparsed = {
            let f = File::parse(&before);
            let edit = AtomEdit { delete: range, insert: replace_with.to_string() };
            f.incremental_reparse(&edit).unwrap()
        };
        assert_eq_text!(
            &dump_tree(fully_reparsed.syntax()),
            &dump_tree(incrementally_reparsed.syntax()),
        )
    }

    do_check(r"
fn foo() {
    let x = foo + <|>bar<|>
}
", "baz");
    do_check(r"
struct Foo {
    f: foo<|><|>
}
", ",\n    g: (),");
    do_check(r"
fn foo {
    let;
    1 + 1;
    <|>92<|>;
}
", "62");
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
    fs::read_to_string(path).unwrap().replace("\r\n", "\n")
}

pub fn dir_tests<F>(paths: &[&str], f: F)
    where
        F: Fn(&str) -> String,
{
    for path in collect_tests(paths) {
        let input_code = read_text(&path);
        let parse_tree = f(&input_code);
        let path = path.with_extension("txt");
        if !path.exists() {
            println!("\nfile: {}", path.display());
            println!("No .txt file with expected result, creating...\n");
            println!("{}\n{}", input_code, parse_tree);
            fs::write(&path, parse_tree).unwrap();
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

fn collect_tests(paths: &[&str]) -> Vec<PathBuf> {
    paths
        .iter()
        .flat_map(|path| {
            let path = test_data_dir().join(path);
            test_from_dir(&path).into_iter()
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
    project_dir().join("crates/libsyntax2/tests/data")
}

fn dump_tokens(tokens: &[libsyntax2::Token], text: &str) -> String {
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
