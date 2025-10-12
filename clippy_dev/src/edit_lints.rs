use crate::parse::cursor::{self, Capture, Cursor};
use crate::parse::{DeprecatedLint, Lint, ParseCx, RenamedLint};
use crate::update_lints::generate_lint_files;
use crate::utils::{
    ErrAction, FileUpdater, UpdateMode, UpdateStatus, Version, delete_dir_if_exists, delete_file_if_exists,
    expect_action, try_rename_dir, try_rename_file, walk_dir_no_dot_or_target,
};
use rustc_lexer::TokenKind;
use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};
use std::{fs, io};

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

    let Some(lint) = lints.iter().find(|l| l.name == name) else {
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

    let mod_path = {
        let mut mod_path = PathBuf::from(format!("clippy_lints/src/{}", lint.module));
        if mod_path.is_dir() {
            mod_path = mod_path.join("mod");
        }

        mod_path.set_extension("rs");
        mod_path
    };

    if remove_lint_declaration(name, &mod_path, &mut lints).unwrap_or(false) {
        generate_lint_files(UpdateMode::Change, &lints, &deprecated_lints, &renamed_lints);
        println!("info: `{name}` has successfully been deprecated");
        println!("note: you must run `cargo uitest` to update the test results");
    } else {
        eprintln!("error: lint not found");
    }
}

pub fn uplift<'cx, 'env: 'cx>(cx: ParseCx<'cx>, clippy_version: Version, old_name: &'env str, new_name: &'env str) {
    let mut lints = cx.find_lint_decls();
    let (deprecated_lints, mut renamed_lints) = cx.read_deprecated_lints();

    let Some(lint) = lints.iter().find(|l| l.name == old_name) else {
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

    let mod_path = {
        let mut mod_path = PathBuf::from(format!("clippy_lints/src/{}", lint.module));
        if mod_path.is_dir() {
            mod_path = mod_path.join("mod");
        }

        mod_path.set_extension("rs");
        mod_path
    };

    if remove_lint_declaration(old_name, &mod_path, &mut lints).unwrap_or(false) {
        generate_lint_files(UpdateMode::Change, &lints, &deprecated_lints, &renamed_lints);
        println!("info: `{old_name}` has successfully been uplifted");
        println!("note: you must run `cargo uitest` to update the test results");
    } else {
        eprintln!("error: lint not found");
    }
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

    // Some tests are named `lint_name_suffix` which should also be renamed,
    // but we can't do that if the renamed lint's name overlaps with another
    // lint. e.g. renaming 'foo' to 'bar' when a lint 'foo_bar' also exists.
    let change_prefixed_tests = lints.get(lint_idx + 1).is_none_or(|l| !l.name.starts_with(old_name));

    let mut mod_edit = ModEdit::None;
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
                mod_edit = ModEdit::Rename;
            }

            lint.module = cx.str_buf.with(|buf| {
                buf.push_str(&lint.module[..lint.module.len() - old_name.len()]);
                buf.push_str(new_name);
                cx.arena.alloc_str(buf)
            });
        }
        rename_test_files(old_name, new_name, change_prefixed_tests);
        lints[lint_idx].name = new_name;
        lints.sort_by(|lhs, rhs| lhs.name.cmp(rhs.name));
    } else {
        println!("Renamed `clippy::{old_name}` to `clippy::{new_name}`");
        println!("Since `{new_name}` already exists the existing code has not been changed");
        return;
    }

    let mut update_fn = file_update_fn(old_name, new_name, mod_edit);
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

fn remove_lint_declaration(name: &str, path: &Path, lints: &mut Vec<Lint<'_>>) -> io::Result<bool> {
    fn remove_lint(name: &str, lints: &mut Vec<Lint<'_>>) {
        lints.iter().position(|l| l.name == name).map(|pos| lints.remove(pos));
    }

    fn remove_test_assets(name: &str) {
        let test_file_stem = format!("tests/ui/{name}");
        let path = Path::new(&test_file_stem);

        // Some lints have their own directories, delete them
        if path.is_dir() {
            let _ = fs::remove_dir_all(path);
            return;
        }

        // Remove all related test files
        let _ = fs::remove_file(path.with_extension("rs"));
        let _ = fs::remove_file(path.with_extension("stderr"));
        let _ = fs::remove_file(path.with_extension("fixed"));
    }

    fn remove_impl_lint_pass(lint_name_upper: &str, content: &mut String) {
        let impl_lint_pass_start = content.find("impl_lint_pass!").unwrap_or_else(|| {
            content
                .find("declare_lint_pass!")
                .unwrap_or_else(|| panic!("failed to find `impl_lint_pass`"))
        });
        let mut impl_lint_pass_end = content[impl_lint_pass_start..]
            .find(']')
            .expect("failed to find `impl_lint_pass` terminator");

        impl_lint_pass_end += impl_lint_pass_start;
        if let Some(lint_name_pos) = content[impl_lint_pass_start..impl_lint_pass_end].find(lint_name_upper) {
            let mut lint_name_end = impl_lint_pass_start + (lint_name_pos + lint_name_upper.len());
            for c in content[lint_name_end..impl_lint_pass_end].chars() {
                // Remove trailing whitespace
                if c == ',' || c.is_whitespace() {
                    lint_name_end += 1;
                } else {
                    break;
                }
            }

            content.replace_range(impl_lint_pass_start + lint_name_pos..lint_name_end, "");
        }
    }

    if path.exists()
        && let Some(lint) = lints.iter().find(|l| l.name == name)
    {
        if lint.module == name {
            // The lint name is the same as the file, we can just delete the entire file
            fs::remove_file(path)?;
        } else {
            // We can't delete the entire file, just remove the declaration

            if let Some(Some("mod.rs")) = path.file_name().map(OsStr::to_str) {
                // Remove clippy_lints/src/some_mod/some_lint.rs
                let mut lint_mod_path = path.to_path_buf();
                lint_mod_path.set_file_name(name);
                lint_mod_path.set_extension("rs");

                let _ = fs::remove_file(lint_mod_path);
            }

            let mut content =
                fs::read_to_string(path).unwrap_or_else(|_| panic!("failed to read `{}`", path.to_string_lossy()));

            eprintln!(
                "warn: you will have to manually remove any code related to `{name}` from `{}`",
                path.display()
            );

            assert!(
                content[lint.declaration_range].contains(&name.to_uppercase()),
                "error: `{}` does not contain lint `{}`'s declaration",
                path.display(),
                lint.name
            );

            // Remove lint declaration (declare_clippy_lint!)
            content.replace_range(lint.declaration_range, "");

            // Remove the module declaration (mod xyz;)
            let mod_decl = format!("\nmod {name};");
            content = content.replacen(&mod_decl, "", 1);

            remove_impl_lint_pass(&lint.name.to_uppercase(), &mut content);
            fs::write(path, content).unwrap_or_else(|_| panic!("failed to write to `{}`", path.to_string_lossy()));
        }

        remove_test_assets(name);
        remove_lint(name, lints);
        return Ok(true);
    }

    Ok(false)
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
                            if !matches!(mod_edit, ModEdit::None)
                                && let Some(pos) = cursor.find_ident(old_name)
                            {
                                match mod_edit {
                                    ModEdit::Rename => {
                                        dst.push_str(&src[copy_pos as usize..pos as usize]);
                                        dst.push_str(new_name);
                                        copy_pos = cursor.pos();
                                        changed = true;
                                    },
                                    ModEdit::Delete if cursor.match_pat(cursor::Pat::Semi) => {
                                        let mut start = &src[copy_pos as usize..match_start as usize];
                                        if start.ends_with("\n\n") {
                                            start = &start[..start.len() - 1];
                                        }
                                        dst.push_str(start);
                                        copy_pos = cursor.pos();
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
