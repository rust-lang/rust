use check_diff::{
    CheckDiffError, CheckDiffRunners, CodeFormatter, FormatCodeError, Repository,
    RustFmtFileFinder, check_diff,
};
use std::fs::File;
use tempfile::Builder;

struct DoNothingFormatter;

impl CodeFormatter for DoNothingFormatter {
    fn format_code(&self, _code: &str) -> Result<String, FormatCodeError> {
        Ok(String::new())
    }
}

/// Formatter that adds a white space to the end of the codd
struct AddWhiteSpaceFormatter;

impl CodeFormatter for AddWhiteSpaceFormatter {
    fn format_code(&self, code: &str) -> Result<String, FormatCodeError> {
        let result = code.to_string() + " ";
        Ok(result)
    }
}

#[test]
fn search_for_files_correctly_non_nested() -> Result<(), Box<dyn std::error::Error>> {
    let dir = Builder::new().tempdir_in("").unwrap();
    let file_path = dir.path().join("test.rs");
    let _tmp_file = File::create(file_path)?;

    let repo = Repository::new("", dir.path());
    let file_finder = RustFmtFileFinder::from_repository(&repo);

    let mut count = 0;
    for _ in file_finder.iter() {
        count += 1;
    }

    assert_eq!(count, 1);

    Ok(())
}

#[test]
fn search_for_files_correctly_nested() -> Result<(), Box<dyn std::error::Error>> {
    let dir = Builder::new().tempdir_in("").unwrap();
    let file_path = dir.path().join("test.rs");
    let _tmp_file = File::create(file_path)?;

    let nested_dir = Builder::new().tempdir_in(dir.path()).unwrap();
    let nested_file_path = nested_dir.path().join("nested.rs");
    let _ = File::create(nested_file_path)?;

    let repo = Repository::new("", dir.path());
    let file_finder = RustFmtFileFinder::from_repository(&repo);

    let mut count = 0;
    for _ in file_finder.iter() {
        count += 1;
    }

    assert_eq!(count, 2);

    Ok(())
}

#[test]
fn check_diff_test_no_formatting_difference() -> Result<(), CheckDiffError> {
    let runners = CheckDiffRunners::new(DoNothingFormatter, DoNothingFormatter);

    let dir = Builder::new().tempdir_in("").unwrap();
    let file_path = dir.path().join("test.rs");
    let _tmp_file = File::create(file_path)?;
    let repo = Repository::new("https://github.com/rust-lang/rustfmt.git", dir);
    let repos = [repo];
    let workers = std::num::NonZeroU8::new(1).unwrap();

    let errors = check_diff(&runners, &repos, workers);
    assert_eq!(errors.len(), 0);
    Ok(())
}

#[test]
fn check_diff_test_formatting_difference() -> Result<(), CheckDiffError> {
    let runners = CheckDiffRunners::new(DoNothingFormatter, AddWhiteSpaceFormatter);
    let dir = Builder::new().tempdir_in("").unwrap();
    let file_path = dir.path().join("test.rs");
    let _tmp_file = File::create(file_path)?;
    let repo = Repository::new("https://github.com/rust-lang/rustfmt.git", dir);
    let repos = [repo];
    let workers = std::num::NonZeroU8::new(1).unwrap();

    let errors = check_diff(&runners, &repos, workers);
    assert_ne!(errors.len(), 0);
    Ok(())
}
