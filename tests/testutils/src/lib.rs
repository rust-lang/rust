extern crate difference;
extern crate file;

use std::path::{PathBuf, Path};
use std::fs::read_dir;

use difference::Changeset;

pub fn dir_tests<F>(
    paths: &[&str],
    f: F
)
where
    F: Fn(&str) -> String
{
    for path in collect_tests(paths) {
        let actual = {
            let text = file::get_text(&path).unwrap();
            f(&text)
        };
        let path = path.with_extension("txt");
        if !path.exists() {
            println!("\nfile: {}", path.display());
            println!("No .txt file with expected result, creating...");
            file::put_text(&path, actual).unwrap();
            panic!("No expected result")
        }
        let expected = file::get_text(&path).unwrap();
        let expected = expected.as_str();
        let actual = actual.as_str();
        assert_equal_text(expected, actual, &path);
    }
}

fn assert_equal_text(
    expected: &str,
    actual: &str,
    path: &Path
) {
    if expected != actual {
        print_difference(expected, actual, path)
    }
}

fn collect_tests(paths: &[&str]) -> Vec<PathBuf> {
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