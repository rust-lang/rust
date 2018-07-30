extern crate difference;
extern crate file;

use std::fs::read_dir;
use std::path::{Path, PathBuf};

use difference::Changeset;

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
    file::get_text(path).unwrap().replace("\r\n", "\n")
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
            file::put_text(&path, parse_tree).unwrap();
            panic!("No expected result")
        }
        let expected = read_text(&path);
        let expected = expected.as_str();
        let parse_tree = parse_tree.as_str();
        assert_equal_text(expected, parse_tree, &path);
    }
}

fn assert_equal_text(expected: &str, actual: &str, path: &Path) {
    if expected != actual {
        print_difference(expected, actual, path)
    }
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
    for file in read_dir(&dir).unwrap() {
        let file = file.unwrap();
        let path = file.path();
        if path.extension().unwrap_or_default() == "rs" {
            acc.push(path);
        }
    }
    acc.sort();
    acc
}

const REWRITE: bool = false;

fn print_difference(expected: &str, actual: &str, path: &Path) {
    let dir = project_dir();
    let path = path.strip_prefix(&dir).unwrap_or_else(|_| path);
    if expected.trim() == actual.trim() {
        println!("whitespace difference, rewriting");
        println!("file: {}\n", path.display());
        file::put_text(path, actual).unwrap();
        return;
    }
    if REWRITE {
        println!("rewriting {}", path.display());
        file::put_text(path, actual).unwrap();
        return;
    }
    let changeset = Changeset::new(actual, expected, "\n");
    print!("{}", changeset);
    println!("file: {}\n", path.display());
    panic!("Comparison failed")
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
    project_dir().join("tests/data")
}
