use crate::parse::cursor::{self, Capture, Cursor};
use crate::parse::{DeprecatedLint, Lint, ParseCx, RenamedLint};
use crate::update_lints::generate_lint_files;
use crate::utils::{
    ErrAction, FileUpdater, UpdateMode, UpdateStatus, Version, delete_dir_if_exists, delete_file_if_exists,
    expect_action, try_rename_dir, try_rename_file, walk_dir_no_dot_or_target,
};
use rustc_lexer::TokenKind;
use std::ffi::OsString;
use std::fs;
use std::path::Path;

/// Runs the `deprecate` command
///
/// This does the following:
/// * Adds an entry to `deprecated_lints.rs`.
/// * Removes the lint declaration (and the entire file if applicable)
///
/// # Panics
///
/// If a file path could not read from or written to
pub fn deprecate<'cx, 'env: 'cx>(cx: ParseCx<'cx>, clippy_version: Version, name: &'env str, reason: &'env str) {
    let mut lints = cx.find_lint_decls();
    let (mut deprecated_lints, renamed_lints) = cx.read_deprecated_lints();

    let Some(lint_idx) = lints.iter().position(|l| l.name == name) else {
        eprintln!("error: failed to find lint `{name}`");
        return;
    };

    let prefixed_name = cx.str_buf.with(|buf| {
        buf.extend(["clippy::", name]);
        cx.arena.alloc_str(buf)
    });
    match deprecated_lints.binary_search_by(|x| x.name.cmp(prefixed_name)) {
        Ok(_) => {
            println!("`{name}` is already deprecated");
            return;
        },
        Err(idx) => deprecated_lints.insert(
            idx,
            DeprecatedLint {
                name: prefixed_name,
                reason,
                version: cx.str_buf.alloc_display(cx.arena, clippy_version.rust_display()),
            },
        ),
    }

    remove_lint_declaration(lint_idx, &mut lints, &mut FileUpdater::default());
    generate_lint_files(UpdateMode::Change, &lints, &deprecated_lints, &renamed_lints);
    println!("info: `{name}` has successfully been deprecated");
    println!("note: you must run `cargo uitest` to update the test results");
}

pub fn uplift<'cx, 'env: 'cx>(cx: ParseCx<'cx>, clippy_version: Version, old_name: &'env str, new_name: &'env str) {
    let mut lints = cx.find_lint_decls();
    let (deprecated_lints, mut renamed_lints) = cx.read_deprecated_lints();

    let Some(lint_idx) = lints.iter().position(|l| l.name == old_name) else {
        eprintln!("error: failed to find lint `{old_name}`");
        return;
    };

    let old_name_prefixed = cx.str_buf.with(|buf| {
        buf.extend(["clippy::", old_name]);
        cx.arena.alloc_str(buf)
    });
    for lint in &mut renamed_lints {
        if lint.new_name == old_name_prefixed {
            lint.new_name = new_name;
        }
    }
    match renamed_lints.binary_search_by(|x| x.old_name.cmp(old_name_prefixed)) {
        Ok(_) => {
            println!("`{old_name}` is already deprecated");
            return;
        },
        Err(idx) => renamed_lints.insert(
            idx,
            RenamedLint {
                old_name: old_name_prefixed,
                new_name,
                version: cx.str_buf.alloc_display(cx.arena, clippy_version.rust_display()),
            },
        ),
    }

    let mut updater = FileUpdater::default();
    let remove_mod = remove_lint_declaration(lint_idx, &mut lints, &mut updater);
    let mut update_fn = uplift_update_fn(old_name, new_name, remove_mod);
    for e in walk_dir_no_dot_or_target(".") {
        let e = expect_action(e, ErrAction::Read, ".");
        if e.path().as_os_str().as_encoded_bytes().ends_with(b".rs") {
            updater.update_file(e.path(), &mut update_fn);
        }
    }
    generate_lint_files(UpdateMode::Change, &lints, &deprecated_lints, &renamed_lints);
    println!("info: `{old_name}` has successfully been uplifted as `{new_name}`");
    println!("note: you must run `cargo uitest` to update the test results");
}

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
pub fn rename<'cx, 'env: 'cx>(cx: ParseCx<'cx>, clippy_version: Version, old_name: &'env str, new_name: &'env str) {
    let mut updater = FileUpdater::default();
    let mut lints = cx.find_lint_decls();
    let (deprecated_lints, mut renamed_lints) = cx.read_deprecated_lints();

    let Ok(lint_idx) = lints.binary_search_by(|x| x.name.cmp(old_name)) else {
        panic!("could not find lint `{old_name}`");
    };

    let old_name_prefixed = cx.str_buf.with(|buf| {
        buf.extend(["clippy::", old_name]);
        cx.arena.alloc_str(buf)
    });
    let new_name_prefixed = cx.str_buf.with(|buf| {
        buf.extend(["clippy::", new_name]);
        cx.arena.alloc_str(buf)
    });

    for lint in &mut renamed_lints {
        if lint.new_name == old_name_prefixed {
            lint.new_name = new_name_prefixed;
        }
    }
    match renamed_lints.binary_search_by(|x| x.old_name.cmp(old_name_prefixed)) {
        Ok(_) => {
            println!("`{old_name}` already has a rename registered");
            return;
        },
        Err(idx) => {
            renamed_lints.insert(
                idx,
                RenamedLint {
                    old_name: old_name_prefixed,
                    new_name: new_name_prefixed,
                    version: cx.str_buf.alloc_display(cx.arena, clippy_version.rust_display()),
                },
            );
        },
    }

    let mut rename_mod = false;
    if lints.binary_search_by(|x| x.name.cmp(new_name)).is_err() {
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
                rename_mod = true;
            }

            lint.module = cx.str_buf.with(|buf| {
                buf.push_str(&lint.module[..lint.module.len() - old_name.len()]);
                buf.push_str(new_name);
                cx.arena.alloc_str(buf)
            });
        }

        rename_test_files(
            old_name,
            new_name,
            &lints[lint_idx + 1..]
                .iter()
                .map(|l| l.name)
                .take_while(|&n| n.starts_with(old_name))
                .collect::<Vec<_>>(),
        );
        lints[lint_idx].name = new_name;
        lints.sort_by(|lhs, rhs| lhs.name.cmp(rhs.name));
    } else {
        println!("Renamed `clippy::{old_name}` to `clippy::{new_name}`");
        println!("Since `{new_name}` already exists the existing code has not been changed");
        return;
    }

    let mut update_fn = rename_update_fn(old_name, new_name, rename_mod);
    for e in walk_dir_no_dot_or_target(".") {
        let e = expect_action(e, ErrAction::Read, ".");
        if e.path().as_os_str().as_encoded_bytes().ends_with(b".rs") {
            updater.update_file(e.path(), &mut update_fn);
        }
    }
    generate_lint_files(UpdateMode::Change, &lints, &deprecated_lints, &renamed_lints);

    println!("Renamed `clippy::{old_name}` to `clippy::{new_name}`");
    println!("All code referencing the old name has been updated");
    println!("Make sure to inspect the results as some things may have been missed");
    println!("note: `cargo uibless` still needs to be run to update the test results");
}

/// Removes a lint's declaration and test files. Returns whether the module containing the
/// lint was deleted.
fn remove_lint_declaration(lint_idx: usize, lints: &mut Vec<Lint<'_>>, updater: &mut FileUpdater) -> bool {
    let lint = lints.remove(lint_idx);
    let delete_mod = if lints.iter().all(|l| l.module != lint.module) {
        delete_file_if_exists(lint.path.as_ref())
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
        false
    };
    delete_test_files(
        lint.name,
        &lints[lint_idx..]
            .iter()
            .map(|l| l.name)
            .take_while(|&n| n.starts_with(lint.name))
            .collect::<Vec<_>>(),
    );

    delete_mod
}

fn collect_ui_test_names(lint: &str, ignored_prefixes: &[&str], dst: &mut Vec<(OsString, bool)>) {
    for e in fs::read_dir("tests/ui").expect("error reading `tests/ui`") {
        let e = e.expect("error reading `tests/ui`");
        let name = e.file_name();
        if name.as_encoded_bytes().starts_with(lint.as_bytes())
            && !ignored_prefixes
                .iter()
                .any(|&pre| name.as_encoded_bytes().starts_with(pre.as_bytes()))
            && let Ok(ty) = e.file_type()
            && (ty.is_file() || ty.is_dir())
        {
            dst.push((name, ty.is_file()));
        }
    }
}

fn collect_ui_toml_test_names(lint: &str, ignored_prefixes: &[&str], dst: &mut Vec<(OsString, bool)>) {
    for e in fs::read_dir("tests/ui-toml").expect("error reading `tests/ui-toml`") {
        let e = e.expect("error reading `tests/ui-toml`");
        let name = e.file_name();
        if name.as_encoded_bytes().starts_with(lint.as_bytes())
            && !ignored_prefixes
                .iter()
                .any(|&pre| name.as_encoded_bytes().starts_with(pre.as_bytes()))
            && e.file_type().is_ok_and(|ty| ty.is_dir())
        {
            dst.push((name, false));
        }
    }
}

/// Renames all test files for the given lint where the file name does not start with any
/// of the given prefixes.
fn rename_test_files(old_name: &str, new_name: &str, ignored_prefixes: &[&str]) {
    let mut tests: Vec<(OsString, bool)> = Vec::new();

    let mut old_buf = OsString::from("tests/ui/");
    let mut new_buf = OsString::from("tests/ui/");
    collect_ui_test_names(old_name, ignored_prefixes, &mut tests);
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
    collect_ui_toml_test_names(old_name, ignored_prefixes, &mut tests);
    for (name, _) in &tests {
        old_buf.push(name);
        new_buf.extend([new_name.as_ref(), name.slice_encoded_bytes(old_name.len()..)]);
        try_rename_dir(old_buf.as_ref(), new_buf.as_ref());
        old_buf.truncate("tests/ui/".len());
        new_buf.truncate("tests/ui/".len());
    }
}

/// Deletes all test files for the given lint where the file name does not start with any
/// of the given prefixes.
fn delete_test_files(lint: &str, ignored_prefixes: &[&str]) {
    let mut tests = Vec::new();

    let mut buf = OsString::from("tests/ui/");
    collect_ui_test_names(lint, ignored_prefixes, &mut tests);
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
    collect_ui_toml_test_names(lint, ignored_prefixes, &mut tests);
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

/// Creates an update function which replaces all instances of `clippy::old_name` with
/// `new_name`.
fn uplift_update_fn<'a>(
    old_name: &'a str,
    new_name: &'a str,
    remove_mod: bool,
) -> impl use<'a> + FnMut(&Path, &str, &mut String) -> UpdateStatus {
    move |_, src, dst| {
        let mut copy_pos = 0u32;
        let mut changed = false;
        let mut cursor = Cursor::new(src);
        while let Some(ident) = cursor.find_any_ident() {
            match cursor.get_text(ident) {
                "mod"
                    if remove_mod && cursor.match_all(&[cursor::Pat::Ident(old_name), cursor::Pat::Semi], &mut []) =>
                {
                    dst.push_str(&src[copy_pos as usize..ident.pos as usize]);
                    dst.push_str(new_name);
                    copy_pos = cursor.pos();
                    if src[copy_pos as usize..].starts_with('\n') {
                        copy_pos += 1;
                    }
                    changed = true;
                },
                "clippy" if cursor.match_all(&[cursor::Pat::DoubleColon, cursor::Pat::Ident(old_name)], &mut []) => {
                    dst.push_str(&src[copy_pos as usize..ident.pos as usize]);
                    dst.push_str(new_name);
                    copy_pos = cursor.pos();
                    changed = true;
                },

                _ => {},
            }
        }
        dst.push_str(&src[copy_pos as usize..]);
        UpdateStatus::from_changed(changed)
    }
}

fn rename_update_fn<'a>(
    old_name: &'a str,
    new_name: &'a str,
    rename_mod: bool,
) -> impl use<'a> + FnMut(&Path, &str, &mut String) -> UpdateStatus {
    let old_name_pascal = snake_to_pascal(old_name);
    let new_name_pascal = snake_to_pascal(new_name);
    let old_name_upper = old_name.to_ascii_uppercase();
    let new_name_upper = new_name.to_ascii_uppercase();
    move |_, src, dst| {
        let mut copy_pos = 0u32;
        let mut changed = false;
        let mut cursor = Cursor::new(src);
        let mut captures = [Capture::EMPTY];
        loop {
            match cursor.peek() {
                TokenKind::Eof => break,
                TokenKind::Ident => {
                    let match_start = cursor.pos();
                    let text = cursor.peek_text();
                    cursor.step();
                    match text {
                        // clippy::line_name or clippy::lint-name
                        "clippy" => {
                            if cursor.match_all(&[cursor::Pat::DoubleColon, cursor::Pat::CaptureIdent], &mut captures)
                                && cursor.get_text(captures[0]) == old_name
                            {
                                dst.push_str(&src[copy_pos as usize..captures[0].pos as usize]);
                                dst.push_str(new_name);
                                copy_pos = cursor.pos();
                                changed = true;
                            }
                        },
                        // mod lint_name
                        "mod" => {
                            if rename_mod && let Some(pos) = cursor.match_ident(old_name) {
                                dst.push_str(&src[copy_pos as usize..pos as usize]);
                                dst.push_str(new_name);
                                copy_pos = cursor.pos();
                                changed = true;
                            }
                        },
                        // lint_name::
                        name if rename_mod && name == old_name => {
                            let name_end = cursor.pos();
                            if cursor.match_pat(cursor::Pat::DoubleColon) {
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
                            copy_pos = cursor.pos();
                            changed = true;
                        },
                    }
                },
                // //~ lint_name
                TokenKind::LineComment { doc_style: None } => {
                    let text = cursor.peek_text();
                    if text.starts_with("//~")
                        && let Some(text) = text.strip_suffix(old_name)
                        && !text.ends_with(|c| matches!(c, 'a'..='z' | 'A'..='Z' | '0'..='9' | '_'))
                    {
                        dst.push_str(&src[copy_pos as usize..cursor.pos() as usize + text.len()]);
                        dst.push_str(new_name);
                        copy_pos = cursor.pos() + cursor.peek_len();
                        changed = true;
                    }
                    cursor.step();
                },
                // ::lint_name
                TokenKind::Colon
                    if cursor.match_all(&[cursor::Pat::DoubleColon, cursor::Pat::CaptureIdent], &mut captures)
                        && cursor.get_text(captures[0]) == old_name =>
                {
                    dst.push_str(&src[copy_pos as usize..captures[0].pos as usize]);
                    dst.push_str(new_name);
                    copy_pos = cursor.pos();
                    changed = true;
                },
                _ => cursor.step(),
            }
        }

        dst.push_str(&src[copy_pos as usize..]);
        UpdateStatus::from_changed(changed)
    }
}
