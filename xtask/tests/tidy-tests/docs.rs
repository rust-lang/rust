use std::{collections::HashMap, fs, io::prelude::*, io::BufReader, path::Path};

use walkdir::{DirEntry, WalkDir};
use xtask::project_root;

fn is_exclude_dir(p: &Path) -> bool {
    // Test hopefully don't really need comments, and for assists we already
    // have special comments which are source of doc tests and user docs.
    let exclude_dirs = ["tests", "test_data", "assists"];
    let mut cur_path = p;
    while let Some(path) = cur_path.parent() {
        if exclude_dirs.iter().any(|dir| path.ends_with(dir)) {
            return true;
        }
        cur_path = path;
    }

    false
}

fn is_exclude_file(d: &DirEntry) -> bool {
    let file_names = ["tests.rs"];

    d.file_name().to_str().map(|f_n| file_names.iter().any(|name| *name == f_n)).unwrap_or(false)
}

fn is_hidden(entry: &DirEntry) -> bool {
    entry.file_name().to_str().map(|s| s.starts_with('.')).unwrap_or(false)
}

#[test]
fn no_docs_comments() {
    let crates = project_root().join("crates");
    let iter = WalkDir::new(crates);
    let mut missing_docs = Vec::new();
    let mut contains_fixme = Vec::new();
    for f in iter.into_iter().filter_entry(|e| !is_hidden(e)) {
        let f = f.unwrap();
        if f.file_type().is_dir() {
            continue;
        }
        if f.path().extension().map(|it| it != "rs").unwrap_or(false) {
            continue;
        }
        if is_exclude_dir(f.path()) {
            continue;
        }
        if is_exclude_file(&f) {
            continue;
        }
        let mut reader = BufReader::new(fs::File::open(f.path()).unwrap());
        let mut line = String::new();
        reader.read_line(&mut line).unwrap();

        if line.starts_with("//!") {
            if line.contains("FIXME") {
                contains_fixme.push(f.path().to_path_buf())
            }
        } else {
            missing_docs.push(f.path().display().to_string());
        }
    }
    if !missing_docs.is_empty() {
        panic!(
            "\nMissing docs strings\n\n\
             modules:\n{}\n\n",
            missing_docs.join("\n")
        )
    }

    let whitelist = [
        "ra_batch",
        "ra_cli",
        "ra_db",
        "ra_hir",
        "ra_hir_expand",
        "ra_ide",
        "ra_lsp_server",
        "ra_mbe",
        "ra_parser",
        "ra_prof",
        "ra_project_model",
        "ra_syntax",
        "ra_text_edit",
        "ra_tt",
        "ra_hir_ty",
    ];

    let mut has_fixmes = whitelist.iter().map(|it| (*it, false)).collect::<HashMap<&str, bool>>();
    'outer: for path in contains_fixme {
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
