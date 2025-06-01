use crate::update_lints::{RenamedLint, find_lint_decls, generate_lint_files, read_deprecated_lints};
use crate::utils::{
    FileUpdater, RustSearcher, Token, UpdateMode, UpdateStatus, Version, delete_dir_if_exists, delete_file_if_exists,
    try_rename_dir, try_rename_file,
};
use rustc_lexer::TokenKind;
use std::ffi::OsString;
use std::fs;
use std::path::Path;
use walkdir::WalkDir;

/// Runs the `rename_lint` command.
///
/// This does the following:
/// * Adds an entry to `renamed_lints.rs`.
/// * Renames all lint attributes to the new name (e.g. `#[allow(clippy::lint_name)]`).
/// * Renames the lint struct to the new name.
/// * Renames the module containing the lint struct to the new name if it shares a name with the
///   lint.
///
/// # Panics
/// Panics for the following conditions:
/// * If a file path could not read from or then written to
/// * If either lint name has a prefix
/// * If `old_name` doesn't name an existing lint.
/// * If `old_name` names a deprecated or renamed lint.
#[expect(clippy::too_many_lines)]
pub fn rename(clippy_version: Version, old_name: &str, new_name: &str, uplift: bool) {
    if let Some((prefix, _)) = old_name.split_once("::") {
        panic!("`{old_name}` should not contain the `{prefix}` prefix");
    }
    if let Some((prefix, _)) = new_name.split_once("::") {
        panic!("`{new_name}` should not contain the `{prefix}` prefix");
    }

    let mut updater = FileUpdater::default();
    let mut lints = find_lint_decls();
    let (deprecated_lints, mut renamed_lints) = read_deprecated_lints();

    let Ok(lint_idx) = lints.binary_search_by(|x| x.name.as_str().cmp(old_name)) else {
        panic!("could not find lint `{old_name}`");
    };
    let lint = &lints[lint_idx];

    let old_name_prefixed = String::from_iter(["clippy::", old_name]);
    let new_name_prefixed = if uplift {
        new_name.to_owned()
    } else {
        String::from_iter(["clippy::", new_name])
    };

    for lint in &mut renamed_lints {
        if lint.new_name == old_name_prefixed {
            lint.new_name.clone_from(&new_name_prefixed);
        }
    }
    match renamed_lints.binary_search_by(|x| x.old_name.cmp(&old_name_prefixed)) {
        Ok(_) => {
            println!("`{old_name}` already has a rename registered");
            return;
        },
        Err(idx) => {
            renamed_lints.insert(
                idx,
                RenamedLint {
                    old_name: old_name_prefixed,
                    new_name: if uplift {
                        new_name.to_owned()
                    } else {
                        String::from_iter(["clippy::", new_name])
                    },
                    version: clippy_version.rust_display().to_string(),
                },
            );
        },
    }

    // Some tests are named `lint_name_suffix` which should also be renamed,
    // but we can't do that if the renamed lint's name overlaps with another
    // lint. e.g. renaming 'foo' to 'bar' when a lint 'foo_bar' also exists.
    let change_prefixed_tests = lints.get(lint_idx + 1).is_none_or(|l| !l.name.starts_with(old_name));

    let mut mod_edit = ModEdit::None;
    if uplift {
        let is_unique_mod = lints[..lint_idx].iter().any(|l| l.module == lint.module)
            || lints[lint_idx + 1..].iter().any(|l| l.module == lint.module);
        if is_unique_mod {
            if delete_file_if_exists(lint.path.as_ref()) {
                mod_edit = ModEdit::Delete;
            }
        } else {
            updater.update_file(&lint.path, &mut |_, src, dst| -> UpdateStatus {
                let mut start = &src[..lint.declaration_range.start];
                if start.ends_with("\n\n") {
                    start = &start[..start.len() - 1];
                }
                let mut end = &src[lint.declaration_range.end..];
                if end.starts_with("\n\n") {
                    end = &end[1..];
                }
                dst.push_str(start);
                dst.push_str(end);
                UpdateStatus::Changed
            });
        }
        delete_test_files(old_name, change_prefixed_tests);
        lints.remove(lint_idx);
    } else if lints.binary_search_by(|x| x.name.as_str().cmp(new_name)).is_err() {
        let lint = &mut lints[lint_idx];
        if lint.module.ends_with(old_name)
            && lint
                .path
                .file_stem()
                .is_some_and(|x| x.as_encoded_bytes() == old_name.as_bytes())
        {
            let mut new_path = lint.path.with_file_name(new_name).into_os_string();
            new_path.push(".rs");
            if try_rename_file(lint.path.as_ref(), new_path.as_ref()) {
                mod_edit = ModEdit::Rename;
            }

            let mod_len = lint.module.len();
            lint.module.truncate(mod_len - old_name.len());
            lint.module.push_str(new_name);
        }
        rename_test_files(old_name, new_name, change_prefixed_tests);
        new_name.clone_into(&mut lints[lint_idx].name);
        lints.sort_by(|lhs, rhs| lhs.name.cmp(&rhs.name));
    } else {
        println!("Renamed `clippy::{old_name}` to `clippy::{new_name}`");
        println!("Since `{new_name}` already exists the existing code has not been changed");
        return;
    }

    let mut update_fn = file_update_fn(old_name, new_name, mod_edit);
    for file in WalkDir::new(".").into_iter().filter_entry(|e| {
        // Skip traversing some of the larger directories.
        e.path()
            .as_os_str()
            .as_encoded_bytes()
            .get(2..)
            .is_none_or(|x| x != "target".as_bytes() && x != ".git".as_bytes())
    }) {
        let file = file.expect("error reading clippy directory");
        if file.path().as_os_str().as_encoded_bytes().ends_with(b".rs") {
            updater.update_file(file.path(), &mut update_fn);
        }
    }
    generate_lint_files(UpdateMode::Change, &lints, &deprecated_lints, &renamed_lints);

    if uplift {
        println!("Uplifted `clippy::{old_name}` as `{new_name}`");
        if matches!(mod_edit, ModEdit::None) {
            println!("Only the rename has been registered, the code will need to be edited manually");
        } else {
            println!("All the lint's code has been deleted");
            println!("Make sure to inspect the results as some things may have been missed");
        }
    } else {
        println!("Renamed `clippy::{old_name}` to `clippy::{new_name}`");
        println!("All code referencing the old name has been updated");
        println!("Make sure to inspect the results as some things may have been missed");
    }
    println!("note: `cargo uibless` still needs to be run to update the test results");
}

#[derive(Clone, Copy)]
enum ModEdit {
    None,
    Delete,
    Rename,
}

fn collect_ui_test_names(lint: &str, rename_prefixed: bool, dst: &mut Vec<(OsString, bool)>) {
    for e in fs::read_dir("tests/ui").expect("error reading `tests/ui`") {
        let e = e.expect("error reading `tests/ui`");
        let name = e.file_name();
        if let Some((name_only, _)) = name.as_encoded_bytes().split_once(|&x| x == b'.') {
            if name_only.starts_with(lint.as_bytes()) && (rename_prefixed || name_only.len() == lint.len()) {
                dst.push((name, true));
            }
        } else if name.as_encoded_bytes().starts_with(lint.as_bytes()) && (rename_prefixed || name.len() == lint.len())
        {
            dst.push((name, false));
        }
    }
}

fn collect_ui_toml_test_names(lint: &str, rename_prefixed: bool, dst: &mut Vec<(OsString, bool)>) {
    if rename_prefixed {
        for e in fs::read_dir("tests/ui-toml").expect("error reading `tests/ui-toml`") {
            let e = e.expect("error reading `tests/ui-toml`");
            let name = e.file_name();
            if name.as_encoded_bytes().starts_with(lint.as_bytes()) && e.file_type().is_ok_and(|ty| ty.is_dir()) {
                dst.push((name, false));
            }
        }
    } else {
        dst.push((lint.into(), false));
    }
}

/// Renames all test files for the given lint.
///
/// If `rename_prefixed` is `true` this will also rename tests which have the lint name as a prefix.
fn rename_test_files(old_name: &str, new_name: &str, rename_prefixed: bool) {
    let mut tests = Vec::new();

    let mut old_buf = OsString::from("tests/ui/");
    let mut new_buf = OsString::from("tests/ui/");
    collect_ui_test_names(old_name, rename_prefixed, &mut tests);
    for &(ref name, is_file) in &tests {
        old_buf.push(name);
        new_buf.extend([new_name.as_ref(), name.slice_encoded_bytes(old_name.len()..)]);
        if is_file {
            try_rename_file(old_buf.as_ref(), new_buf.as_ref());
        } else {
            try_rename_dir(old_buf.as_ref(), new_buf.as_ref());
        }
        old_buf.truncate("tests/ui/".len());
        new_buf.truncate("tests/ui/".len());
    }

    tests.clear();
    old_buf.truncate("tests/ui".len());
    new_buf.truncate("tests/ui".len());
    old_buf.push("-toml/");
    new_buf.push("-toml/");
    collect_ui_toml_test_names(old_name, rename_prefixed, &mut tests);
    for (name, _) in &tests {
        old_buf.push(name);
        new_buf.extend([new_name.as_ref(), name.slice_encoded_bytes(old_name.len()..)]);
        try_rename_dir(old_buf.as_ref(), new_buf.as_ref());
        old_buf.truncate("tests/ui/".len());
        new_buf.truncate("tests/ui/".len());
    }
}

fn delete_test_files(lint: &str, rename_prefixed: bool) {
    let mut tests = Vec::new();

    let mut buf = OsString::from("tests/ui/");
    collect_ui_test_names(lint, rename_prefixed, &mut tests);
    for &(ref name, is_file) in &tests {
        buf.push(name);
        if is_file {
            delete_file_if_exists(buf.as_ref());
        } else {
            delete_dir_if_exists(buf.as_ref());
        }
        buf.truncate("tests/ui/".len());
    }

    buf.truncate("tests/ui".len());
    buf.push("-toml/");

    tests.clear();
    collect_ui_toml_test_names(lint, rename_prefixed, &mut tests);
    for (name, _) in &tests {
        buf.push(name);
        delete_dir_if_exists(buf.as_ref());
        buf.truncate("tests/ui/".len());
    }
}

fn snake_to_pascal(s: &str) -> String {
    let mut dst = Vec::with_capacity(s.len());
    let mut iter = s.bytes();
    || -> Option<()> {
        dst.push(iter.next()?.to_ascii_uppercase());
        while let Some(c) = iter.next() {
            if c == b'_' {
                dst.push(iter.next()?.to_ascii_uppercase());
            } else {
                dst.push(c);
            }
        }
        Some(())
    }();
    String::from_utf8(dst).unwrap()
}

#[expect(clippy::too_many_lines)]
fn file_update_fn<'a, 'b>(
    old_name: &'a str,
    new_name: &'b str,
    mod_edit: ModEdit,
) -> impl use<'a, 'b> + FnMut(&Path, &str, &mut String) -> UpdateStatus {
    let old_name_pascal = snake_to_pascal(old_name);
    let new_name_pascal = snake_to_pascal(new_name);
    let old_name_upper = old_name.to_ascii_uppercase();
    let new_name_upper = new_name.to_ascii_uppercase();
    move |_, src, dst| {
        let mut copy_pos = 0u32;
        let mut changed = false;
        let mut searcher = RustSearcher::new(src);
        let mut capture = "";
        loop {
            match searcher.peek() {
                TokenKind::Eof => break,
                TokenKind::Ident => {
                    let match_start = searcher.pos();
                    let text = searcher.peek_text();
                    searcher.step();
                    match text {
                        // clippy::line_name or clippy::lint-name
                        "clippy" => {
                            if searcher.match_tokens(&[Token::DoubleColon, Token::CaptureIdent], &mut [&mut capture])
                                && capture == old_name
                            {
                                dst.push_str(&src[copy_pos as usize..searcher.pos() as usize - capture.len()]);
                                dst.push_str(new_name);
                                copy_pos = searcher.pos();
                                changed = true;
                            }
                        },
                        // mod lint_name
                        "mod" => {
                            if !matches!(mod_edit, ModEdit::None)
                                && searcher.match_tokens(&[Token::CaptureIdent], &mut [&mut capture])
                                && capture == old_name
                            {
                                match mod_edit {
                                    ModEdit::Rename => {
                                        dst.push_str(&src[copy_pos as usize..searcher.pos() as usize - capture.len()]);
                                        dst.push_str(new_name);
                                        copy_pos = searcher.pos();
                                        changed = true;
                                    },
                                    ModEdit::Delete if searcher.match_tokens(&[Token::Semi], &mut []) => {
                                        let mut start = &src[copy_pos as usize..match_start as usize];
                                        if start.ends_with("\n\n") {
                                            start = &start[..start.len() - 1];
                                        }
                                        dst.push_str(start);
                                        copy_pos = searcher.pos();
                                        if src[copy_pos as usize..].starts_with("\n\n") {
                                            copy_pos += 1;
                                        }
                                        changed = true;
                                    },
                                    ModEdit::Delete | ModEdit::None => {},
                                }
                            }
                        },
                        // lint_name::
                        name if matches!(mod_edit, ModEdit::Rename) && name == old_name => {
                            let name_end = searcher.pos();
                            if searcher.match_tokens(&[Token::DoubleColon], &mut []) {
                                dst.push_str(&src[copy_pos as usize..match_start as usize]);
                                dst.push_str(new_name);
                                copy_pos = name_end;
                                changed = true;
                            }
                        },
                        // LINT_NAME or LintName
                        name => {
                            let replacement = if name == old_name_upper {
                                &new_name_upper
                            } else if name == old_name_pascal {
                                &new_name_pascal
                            } else {
                                continue;
                            };
                            dst.push_str(&src[copy_pos as usize..match_start as usize]);
                            dst.push_str(replacement);
                            copy_pos = searcher.pos();
                            changed = true;
                        },
                    }
                },
                // //~ lint_name
                TokenKind::LineComment { doc_style: None } => {
                    let text = searcher.peek_text();
                    if text.starts_with("//~")
                        && let Some(text) = text.strip_suffix(old_name)
                        && !text.ends_with(|c| matches!(c, 'a'..='z' | 'A'..='Z' | '0'..='9' | '_'))
                    {
                        dst.push_str(&src[copy_pos as usize..searcher.pos() as usize + text.len()]);
                        dst.push_str(new_name);
                        copy_pos = searcher.pos() + searcher.peek_len();
                        changed = true;
                    }
                    searcher.step();
                },
                // ::lint_name
                TokenKind::Colon
                    if searcher.match_tokens(&[Token::DoubleColon, Token::CaptureIdent], &mut [&mut capture])
                        && capture == old_name =>
                {
                    dst.push_str(&src[copy_pos as usize..searcher.pos() as usize - capture.len()]);
                    dst.push_str(new_name);
                    copy_pos = searcher.pos();
                    changed = true;
                },
                _ => searcher.step(),
            }
        }

        dst.push_str(&src[copy_pos as usize..]);
        UpdateStatus::from_changed(changed)
    }
}
