use regex::Regex;
use similar::TextDiff;
use std::path::Path;

#[cfg(test)]
mod tests;

pub fn diff() -> Diff {
    Diff::new()
}

#[derive(Debug)]
pub struct Diff {
    expected: Option<String>,
    expected_name: Option<String>,
    actual: Option<String>,
    actual_name: Option<String>,
    normalizers: Vec<(String, String)>,
}

impl Diff {
    /// Construct a bare `diff` invocation.
    pub fn new() -> Self {
        Self {
            expected: None,
            expected_name: None,
            actual: None,
            actual_name: None,
            normalizers: Vec::new(),
        }
    }

    /// Specify the expected output for the diff from a file.
    pub fn expected_file<P: AsRef<Path>>(&mut self, path: P) -> &mut Self {
        let path = path.as_ref();
        let content = std::fs::read_to_string(path).expect("failed to read file");
        let name = path.to_string_lossy().to_string();

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
        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(e) => panic!("failed to read `{}`: {:?}", path.display(), e),
        };
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

    /// Executes the diff process, prints any differences to the standard error.
    #[track_caller]
    pub fn run(&self) {
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
            panic!(
                "test failed: `{}` is different from `{}`\n\n{}",
                expected_name, actual_name, output
            )
        }
    }
}
