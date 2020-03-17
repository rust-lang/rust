mod cli;

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

use walkdir::{DirEntry, WalkDir};
use xtask::{not_bash::fs2, project_root};

#[test]
fn rust_files_are_tidy() {
    let mut tidy_docs = TidyDocs::default();
    for path in rust_files() {
        let text = fs2::read_to_string(&path).unwrap();
        check_todo(&path, &text);
        check_trailing_ws(&path, &text);
        tidy_docs.visit(&path, &text);
    }
    tidy_docs.finish();
}

fn check_todo(path: &Path, text: &str) {
    if path.ends_with("tests/cli.rs") {
        return;
    }
    if text.contains("TODO") || text.contains("TOOD") || text.contains("todo!") {
        panic!(
            "\nTODO markers should not be committed to the master branch,\n\
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
    for line in text.lines() {
        if line.chars().last().map(char::is_whitespace) == Some(true) {
            panic!("Trailing whitespace in {}", path.display())
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
        if is_exclude_dir(path, &["tests", "test_data", "handlers"]) {
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
                self.contains_fixme.push(path.to_path_buf())
            }
        } else {
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
            "ra_db",
            "ra_hir",
            "ra_hir_expand",
            "ra_ide",
            "ra_mbe",
            "ra_parser",
            "ra_prof",
            "ra_project_model",
            "ra_syntax",
            "ra_text_edit",
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
    let mut cur_path = p;
    while let Some(path) = cur_path.parent() {
        if dirs_to_exclude.iter().any(|dir| path.ends_with(dir)) {
            return true;
        }
        cur_path = path;
    }

    false
}

fn rust_files() -> impl Iterator<Item = PathBuf> {
    let crates = project_root().join("crates");
    let iter = WalkDir::new(crates);
    return iter
        .into_iter()
        .filter_entry(|e| !is_hidden(e))
        .map(|e| e.unwrap())
        .filter(|e| !e.file_type().is_dir())
        .map(|e| e.into_path())
        .filter(|path| path.extension().map(|it| it == "rs").unwrap_or(false));

    fn is_hidden(entry: &DirEntry) -> bool {
        entry.file_name().to_str().map(|s| s.starts_with('.')).unwrap_or(false)
    }
}
