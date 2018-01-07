extern crate difference;
extern crate file;

use std::path::{PathBuf, Path};
use std::fs::read_dir;

use difference::Changeset;

pub fn assert_equal_text(
    expected: &str,
    actual: &str,
    path: &Path
) {
    if expected != actual {
        print_difference(expected, actual, path)
    }
}

pub fn collect_tests(paths: &[&str]) -> Vec<PathBuf> {
    paths.iter().flat_map(|path|  {
        let path = test_data_dir().join(path);
        test_from_dir(&path).into_iter()
    }).collect()
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

fn print_difference(expected: &str, actual: &str, path: &Path) {
    let dir = project_dir();
    let path = path.strip_prefix(&dir).unwrap_or_else(|_| path);
    println!("\nfile: {}", path.display());
    if expected.trim() == actual.trim() {
        println!("whitespace difference");
        println!("rewriting the file");
        file::put_text(path, actual).unwrap();
    } else {
        let changeset = Changeset::new(actual, expected, "\n");
        println!("{}", changeset);
    }
    panic!("Comparison failed")
}

fn project_dir() -> PathBuf {
    let dir = env!("CARGO_MANIFEST_DIR");
    PathBuf::from(dir)
        .parent().unwrap()
        .parent().unwrap()
        .to_owned()
}

fn test_data_dir() -> PathBuf {
    project_dir().join("tests/data")
}