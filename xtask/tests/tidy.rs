use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

use xtask::{
    codegen::{self, Mode},
    not_bash::{fs2, run},
    project_root, run_rustfmt, rust_files,
};

#[test]
fn generated_grammar_is_fresh() {
    if let Err(error) = codegen::generate_syntax(Mode::Verify) {
        panic!("{}. Please update it by running `cargo xtask codegen`", error);
    }
}

#[test]
fn generated_tests_are_fresh() {
    if let Err(error) = codegen::generate_parser_tests(Mode::Verify) {
        panic!("{}. Please update tests by running `cargo xtask codegen`", error);
    }
}

#[test]
fn generated_assists_are_fresh() {
    if let Err(error) = codegen::generate_assists_tests(Mode::Verify) {
        panic!("{}. Please update assists by running `cargo xtask codegen`", error);
    }
}

#[test]
fn check_code_formatting() {
    if let Err(error) = run_rustfmt(Mode::Verify) {
        panic!("{}. Please format the code by running `cargo format`", error);
    }
}

#[test]
fn rust_files_are_tidy() {
    let mut tidy_docs = TidyDocs::default();
    for path in rust_files(&project_root().join("crates")) {
        let text = fs2::read_to_string(&path).unwrap();
        check_todo(&path, &text);
        check_trailing_ws(&path, &text);
        deny_clippy(&path, &text);
        tidy_docs.visit(&path, &text);
    }
    tidy_docs.finish();
}

fn deny_clippy(path: &PathBuf, text: &String) {
    if text.contains("[\u{61}llow(clippy") {
        panic!(
            "\n\nallowing lints is forbidden: {}.
rust-analyzer intentionally doesn't check clippy on CI.
You can allow lint globally via `xtask clippy`.
See https://github.com/rust-lang/rust-clippy/issues/5537 for discussion.

",
            path.display()
        )
    }
}

#[test]
fn check_licenses() {
    let expected = "
0BSD OR MIT OR Apache-2.0
Apache-2.0 OR BSL-1.0
Apache-2.0 OR MIT
Apache-2.0/MIT
BSD-2-Clause
BSD-3-Clause
CC0-1.0
ISC
MIT
MIT / Apache-2.0
MIT OR Apache-2.0
MIT/Apache-2.0
MIT/Apache-2.0 AND BSD-2-Clause
Unlicense OR MIT
Unlicense/MIT
Zlib OR Apache-2.0 OR MIT
"
    .lines()
    .filter(|it| !it.is_empty())
    .collect::<Vec<_>>();

    let meta = run!("cargo metadata --format-version 1"; echo = false).unwrap();
    let mut licenses = meta
        .split(|c| c == ',' || c == '{' || c == '}')
        .filter(|it| it.contains(r#""license""#))
        .map(|it| it.trim())
        .map(|it| it[r#""license":"#.len()..].trim_matches('"'))
        .collect::<Vec<_>>();
    licenses.sort();
    licenses.dedup();
    assert_eq!(licenses, expected);
}

fn check_todo(path: &Path, text: &str) {
    let need_todo = &[
        // This file itself obviously needs to use todo (<- like this!).
        "tests/cli.rs",
        // Some of our assists generate `todo!()`.
        "tests/generated.rs",
        "handlers/add_missing_impl_members.rs",
        "handlers/add_turbo_fish.rs",
        "handlers/generate_function.rs",
        // To support generating `todo!()` in assists, we have `expr_todo()` in
        // `ast::make`.
        "ast/make.rs",
        // The documentation in string literals may contain anything for its own purposes
        "completion/generated_features.rs",
    ];
    if need_todo.iter().any(|p| path.ends_with(p)) {
        return;
    }
    if text.contains("TODO") || text.contains("TOOD") || text.contains("todo!") {
        panic!(
            "\nTODO markers or todo! macros should not be committed to the master branch,\n\
             use FIXME instead\n\
             {}\n",
            path.display(),
        )
    }
}

fn check_trailing_ws(path: &Path, text: &str) {
    if is_exclude_dir(path, &["test_data"]) {
        return;
    }
    for (line_number, line) in text.lines().enumerate() {
        if line.chars().last().map(char::is_whitespace) == Some(true) {
            panic!("Trailing whitespace in {} at line {}", path.display(), line_number)
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
        // Test hopefully don't really need comments, and for assists we already
        // have special comments which are source of doc tests and user docs.
        if is_exclude_dir(path, &["tests", "test_data"]) {
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
            if text.contains("// Feature:") || text.contains("// Assist:") {
                return;
            }
            self.missing_docs.push(path.display().to_string());
        }

        fn is_exclude_file(d: &Path) -> bool {
            let file_names = ["tests.rs"];

            d.file_name()
                .unwrap_or_default()
                .to_str()
                .map(|f_n| file_names.iter().any(|name| *name == f_n))
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

        let poorly_documented = [
            "hir",
            "hir_expand",
            "ide",
            "mbe",
            "parser",
            "profile",
            "project_model",
            "syntax",
            "tt",
            "hir_ty",
        ];

        let mut has_fixmes =
            poorly_documented.iter().map(|it| (*it, false)).collect::<HashMap<&str, bool>>();
        'outer: for path in self.contains_fixme {
            for krate in poorly_documented.iter() {
                if path.components().any(|it| it.as_os_str() == *krate) {
                    has_fixmes.insert(krate, true);
                    continue 'outer;
                }
            }
            panic!("FIXME doc in a fully-documented crate: {}", path.display())
        }

        for (krate, has_fixme) in has_fixmes.iter() {
            if !has_fixme {
                panic!("crate {} is fully documented :tada:, remove it from the list of poorly documented crates", krate)
            }
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
