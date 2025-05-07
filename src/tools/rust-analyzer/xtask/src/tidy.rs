#![allow(clippy::disallowed_types, clippy::print_stderr)]
use std::{
    collections::HashSet,
    path::{Path, PathBuf},
};

use itertools::Itertools;
use xshell::Shell;

use xshell::cmd;

use crate::{flags::Tidy, project_root, util::list_files};

impl Tidy {
    pub(crate) fn run(&self, sh: &Shell) -> anyhow::Result<()> {
        check_lsp_extensions_docs(sh);
        files_are_tidy(sh);
        check_licenses(sh);
        Ok(())
    }
}

fn check_lsp_extensions_docs(sh: &Shell) {
    let expected_hash = {
        let lsp_ext_rs =
            sh.read_file(project_root().join("crates/rust-analyzer/src/lsp/ext.rs")).unwrap();
        stable_hash(lsp_ext_rs.as_str())
    };

    let actual_hash = {
        let lsp_extensions_md = sh
            .read_file(project_root().join("docs/book/src/contributing/lsp-extensions.md"))
            .unwrap();
        let text = lsp_extensions_md
            .lines()
            .find_map(|line| line.strip_prefix("lsp/ext.rs hash:"))
            .unwrap()
            .trim();
        u64::from_str_radix(text, 16).unwrap()
    };

    if actual_hash != expected_hash {
        panic!(
            "
lsp/ext.rs was changed without touching lsp-extensions.md.

Expected hash: {expected_hash:x}
Actual hash:   {actual_hash:x}

Please adjust docs/book/src/contributing/lsp-extensions.md.
"
        )
    }
}

fn files_are_tidy(sh: &Shell) {
    let files = list_files(&project_root().join("crates"));

    let mut tidy_docs = TidyDocs::default();
    let mut tidy_marks = TidyMarks::default();
    for path in files {
        let extension = path.extension().unwrap_or_default().to_str().unwrap_or_default();
        match extension {
            "rs" => {
                let text = sh.read_file(&path).unwrap();
                check_test_attrs(&path, &text);
                check_trailing_ws(&path, &text);
                tidy_docs.visit(&path, &text);
                tidy_marks.visit(&path, &text);
            }
            "toml" => {
                let text = sh.read_file(&path).unwrap();
                check_cargo_toml(&path, text);
            }
            _ => (),
        }
    }

    tidy_docs.finish();
    tidy_marks.finish();
}

fn check_cargo_toml(path: &Path, text: String) {
    let mut section = None;
    for (line_no, text) in text.lines().enumerate() {
        let text = text.trim();
        if text.starts_with('[') {
            if !text.ends_with(']') {
                panic!(
                    "\nplease don't add comments or trailing whitespace in section lines.\n\
                        {}:{}\n",
                    path.display(),
                    line_no + 1
                )
            }
            section = Some(text);
            continue;
        }
        let text: String = text.split_whitespace().collect();
        if !text.contains("path=") {
            continue;
        }
        match section {
            Some(s) if s.contains("dev-dependencies") => {
                if text.contains("version") {
                    panic!(
                        "\ncargo internal dev-dependencies should not have a version.\n\
                        {}:{}\n",
                        path.display(),
                        line_no + 1
                    );
                }
            }
            Some(s) if s.contains("dependencies") => {
                if !text.contains("version") {
                    panic!(
                        "\ncargo internal dependencies should have a version.\n\
                        {}:{}\n",
                        path.display(),
                        line_no + 1
                    );
                }
            }
            _ => {}
        }
    }
}

fn check_licenses(sh: &Shell) {
    const EXPECTED: [&str; 20] = [
        "(MIT OR Apache-2.0) AND Unicode-3.0",
        "0BSD OR MIT OR Apache-2.0",
        "Apache-2.0",
        "Apache-2.0 OR BSL-1.0",
        "Apache-2.0 OR MIT",
        "Apache-2.0 WITH LLVM-exception",
        "Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT",
        "Apache-2.0/MIT",
        "CC0-1.0",
        "ISC",
        "MIT",
        "MIT / Apache-2.0",
        "MIT OR Apache-2.0",
        "MIT OR Zlib OR Apache-2.0",
        "MIT/Apache-2.0",
        "MPL-2.0",
        "Unicode-3.0",
        "Unlicense OR MIT",
        "Unlicense/MIT",
        "Zlib",
    ];

    let meta = cmd!(sh, "cargo metadata --format-version 1").read().unwrap();
    let mut licenses = meta
        .split([',', '{', '}'])
        .filter(|it| it.contains(r#""license""#))
        .map(|it| it.trim())
        .map(|it| it[r#""license":"#.len()..].trim_matches('"'))
        .collect::<Vec<_>>();
    licenses.sort_unstable();
    licenses.dedup();
    if licenses != EXPECTED {
        let mut diff = String::new();

        diff.push_str("New Licenses:\n");
        for &l in licenses.iter() {
            if !EXPECTED.contains(&l) {
                diff += &format!("  {l}\n")
            }
        }

        diff.push_str("\nMissing Licenses:\n");
        for l in EXPECTED {
            if !licenses.contains(&l) {
                diff += &format!("  {l}\n")
            }
        }

        panic!("different set of licenses!\n{diff}");
    }
    assert_eq!(licenses, EXPECTED);
}

fn check_test_attrs(path: &Path, text: &str) {
    let panic_rule = "https://github.com/rust-lang/rust-analyzer/blob/master/docs/book/src/contributing/style.md#should_panic";
    let need_panic: &[&str] = &[
        // This file.
        "slow-tests/tidy.rs",
        "test-utils/src/fixture.rs",
        // Generated code from lints contains doc tests in string literals.
        "ide-db/src/generated/lints.rs",
    ];
    if need_panic.iter().any(|p| path.ends_with(p)) {
        return;
    }
    if let Some((line, _)) = text
        .lines()
        .tuple_windows()
        .enumerate()
        .find(|(_, (a, b))| b.contains("#[should_panic") && !a.contains("FIXME"))
    {
        panic!(
            "\ndon't add `#[should_panic]` tests, see:\n\n    {}\n\n   {}:{line}\n",
            panic_rule,
            path.display(),
        )
    }
}

fn check_trailing_ws(path: &Path, text: &str) {
    if is_exclude_dir(path, &["test_data"]) {
        return;
    }
    for (line_number, line) in text.lines().enumerate() {
        if line.chars().last().is_some_and(char::is_whitespace) {
            panic!("Trailing whitespace in {} at line {}", path.display(), line_number + 1)
        }
    }
}

#[derive(Default)]
struct TidyDocs {
    missing_docs: Vec<String>,
    contains_fixme: Vec<PathBuf>,
}

impl TidyDocs {
    fn visit(&mut self, path: &Path, text: &str) {
        // Tests and diagnostic fixes don't need module level comments.
        if is_exclude_dir(path, &["tests", "test_data", "fixes", "grammar", "stdx"]) {
            return;
        }

        if is_exclude_file(path) {
            return;
        }

        let first_line = match text.lines().next() {
            Some(it) => it,
            None => return,
        };

        if first_line.starts_with("//!") {
            if first_line.contains("FIXME") {
                self.contains_fixme.push(path.to_path_buf());
            }
        } else {
            if text.contains("// Feature:")
                || text.contains("// Assist:")
                || text.contains("// Diagnostic:")
            {
                return;
            }
            self.missing_docs.push(path.display().to_string());
        }

        fn is_exclude_file(d: &Path) -> bool {
            let file_names = ["tests.rs", "famous_defs_fixture.rs"];

            d.file_name()
                .unwrap_or_default()
                .to_str()
                .map(|f_n| file_names.contains(&f_n))
                .unwrap_or(false)
        }
    }

    fn finish(self) {
        if !self.missing_docs.is_empty() {
            panic!(
                "\nMissing docs strings\n\n\
                 modules:\n{}\n\n",
                self.missing_docs.join("\n")
            )
        }

        if let Some(path) = self.contains_fixme.first() {
            panic!("FIXME doc in a fully-documented crate: {}", path.display())
        }
    }
}

fn is_exclude_dir(p: &Path, dirs_to_exclude: &[&str]) -> bool {
    p.strip_prefix(project_root())
        .unwrap()
        .components()
        .rev()
        .skip(1)
        .filter_map(|it| it.as_os_str().to_str())
        .any(|it| dirs_to_exclude.contains(&it))
}

#[derive(Default)]
struct TidyMarks {
    hits: HashSet<String>,
    checks: HashSet<String>,
}

impl TidyMarks {
    fn visit(&mut self, _path: &Path, text: &str) {
        find_marks(&mut self.hits, text, "hit");
        find_marks(&mut self.checks, text, "check");
        find_marks(&mut self.checks, text, "check_count");
    }

    fn finish(self) {
        assert!(!self.hits.is_empty());

        let diff: Vec<_> =
            self.hits.symmetric_difference(&self.checks).map(|it| it.as_str()).collect();

        if !diff.is_empty() {
            panic!("unpaired marks: {diff:?}")
        }
    }
}

#[allow(deprecated)]
fn stable_hash(text: &str) -> u64 {
    use std::hash::{Hash, Hasher, SipHasher};

    let text = text.replace('\r', "");
    let mut hasher = SipHasher::default();
    text.hash(&mut hasher);
    hasher.finish()
}

fn find_marks(set: &mut HashSet<String>, text: &str, mark: &str) {
    let mut text = text;
    let mut prev_text = "";
    while text != prev_text {
        prev_text = text;
        if let Some(idx) = text.find(mark) {
            text = &text[idx + mark.len()..];
            if let Some(stripped_text) = text.strip_prefix("!(") {
                text = stripped_text.trim_start();
                if let Some(idx2) = text.find(|c: char| !(c.is_alphanumeric() || c == '_')) {
                    let mark_text = &text[..idx2];
                    set.insert(mark_text.to_owned());
                    text = &text[idx2..];
                }
            }
        }
    }
}

#[test]
fn test() {
    Tidy {}.run(&Shell::new().unwrap()).unwrap();
}
