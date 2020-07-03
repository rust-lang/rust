use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

use xtask::{
    codegen::{self, Mode},
    not_bash::fs2,
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
        tidy_docs.visit(&path, &text);
    }
    tidy_docs.finish();
}

fn check_todo(path: &Path, text: &str) {
    let whitelist = &[
        // This file itself is whitelisted since this test itself contains matches.
        "tests/cli.rs",
        // Some of our assists generate `todo!()` so those files are whitelisted.
        "tests/generated.rs",
        "handlers/add_missing_impl_members.rs",
        "handlers/add_turbo_fish.rs",
        "handlers/generate_function.rs",
        // To support generating `todo!()` in assists, we have `expr_todo()` in ast::make.
        "ast/make.rs",
    ];
    if whitelist.iter().any(|p| path.ends_with(p)) {
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

        let whitelist = [
            "ra_hir",
            "ra_hir_expand",
            "ra_ide",
            "ra_mbe",
            "ra_parser",
            "ra_prof",
            "ra_project_model",
            "ra_syntax",
            "ra_tt",
            "ra_hir_ty",
        ];

        let mut has_fixmes =
            whitelist.iter().map(|it| (*it, false)).collect::<HashMap<&str, bool>>();
        'outer: for path in self.contains_fixme {
            for krate in whitelist.iter() {
                if path.components().any(|it| it.as_os_str() == *krate) {
                    has_fixmes.insert(krate, true);
                    continue 'outer;
                }
            }
            panic!("FIXME doc in a fully-documented crate: {}", path.display())
        }

        for (krate, has_fixme) in has_fixmes.iter() {
            if !has_fixme {
                panic!("crate {} is fully documented, remove it from the white list", krate)
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
