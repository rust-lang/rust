use regex::Regex;
use similar::TextDiff;
use std::path::{Path, PathBuf};

use crate::drop_bomb::DropBomb;
use crate::fs_wrapper;

#[cfg(test)]
mod tests;

#[track_caller]
pub fn diff() -> Diff {
    Diff::new()
}

#[derive(Debug)]
#[must_use]
pub struct Diff {
    expected: Option<String>,
    expected_name: Option<String>,
    expected_file: Option<PathBuf>,
    actual: Option<String>,
    actual_name: Option<String>,
    normalizers: Vec<(String, String)>,
    drop_bomb: DropBomb,
}

impl Diff {
    /// Construct a bare `diff` invocation.
    #[track_caller]
    pub fn new() -> Self {
        Self {
            expected: None,
            expected_name: None,
            expected_file: None,
            actual: None,
            actual_name: None,
            normalizers: Vec::new(),
            drop_bomb: DropBomb::arm("diff"),
        }
    }

    /// Specify the expected output for the diff from a file.
    pub fn expected_file<P: AsRef<Path>>(&mut self, path: P) -> &mut Self {
        let path = path.as_ref();
        let content = fs_wrapper::read_to_string(path);
        let name = path.to_string_lossy().to_string();

        self.expected_file = Some(path.into());
        self.expected = Some(content);
        self.expected_name = Some(name);
        self
    }

    /// Specify the expected output for the diff from a given text string.
    pub fn expected_text<T: AsRef<[u8]>>(&mut self, name: &str, text: T) -> &mut Self {
        self.expected = Some(String::from_utf8_lossy(text.as_ref()).to_string());
        self.expected_name = Some(name.to_string());
        self
    }

    /// Specify the actual output for the diff from a file.
    pub fn actual_file<P: AsRef<Path>>(&mut self, path: P) -> &mut Self {
        let path = path.as_ref();
        let content = fs_wrapper::read_to_string(path);
        let name = path.to_string_lossy().to_string();

        self.actual = Some(content);
        self.actual_name = Some(name);
        self
    }

    /// Specify the actual output for the diff from a given text string.
    pub fn actual_text<T: AsRef<[u8]>>(&mut self, name: &str, text: T) -> &mut Self {
        self.actual = Some(String::from_utf8_lossy(text.as_ref()).to_string());
        self.actual_name = Some(name.to_string());
        self
    }

    /// Specify a regex that should replace text in the "actual" text that will be compared.
    pub fn normalize<R: Into<String>, I: Into<String>>(
        &mut self,
        regex: R,
        replacement: I,
    ) -> &mut Self {
        self.normalizers.push((regex.into(), replacement.into()));
        self
    }

    #[track_caller]
    pub fn run(&mut self) {
        self.drop_bomb.defuse();
        let expected = self.expected.as_ref().expect("expected text not set");
        let mut actual = self.actual.as_ref().expect("actual text not set").to_string();
        let expected_name = self.expected_name.as_ref().unwrap();
        let actual_name = self.actual_name.as_ref().unwrap();
        for (regex, replacement) in &self.normalizers {
            let re = Regex::new(regex).expect("bad regex in custom normalization rule");
            actual = re.replace_all(&actual, replacement).into_owned();
        }

        let output = TextDiff::from_lines(expected, &actual)
            .unified_diff()
            .header(expected_name, actual_name)
            .to_string();

        if !output.is_empty() {
            // If we can bless (meaning we have a file to write into and the `RUSTC_BLESS_TEST`
            // environment variable set), then we write into the file and return.
            if let Some(ref expected_file) = self.expected_file {
                if std::env::var("RUSTC_BLESS_TEST").is_ok() {
                    println!("Blessing `{}`", expected_file.display());
                    fs_wrapper::write(expected_file, actual);
                    return;
                }
            }
            panic!(
                "test failed: `{}` is different from `{}`\n\n{}",
                expected_name, actual_name, output
            )
        }
    }
}
