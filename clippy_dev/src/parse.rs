pub mod cursor;

use self::cursor::{Capture, Cursor};
use crate::utils::{ErrAction, File, expect_action};
use core::range::Range;
use std::fs;
use std::path::{Path, PathBuf};
use walkdir::{DirEntry, WalkDir};

pub struct Lint {
    pub name: String,
    pub group: String,
    pub module: String,
    pub path: PathBuf,
    pub declaration_range: Range<usize>,
}

pub struct DeprecatedLint {
    pub name: String,
    pub reason: String,
    pub version: String,
}

pub struct RenamedLint {
    pub old_name: String,
    pub new_name: String,
    pub version: String,
}

/// Finds all lint declarations (`declare_clippy_lint!`)
#[must_use]
pub fn find_lint_decls() -> Vec<Lint> {
    let mut lints = Vec::with_capacity(1000);
    let mut contents = String::new();
    for e in expect_action(fs::read_dir("."), ErrAction::Read, ".") {
        let e = expect_action(e, ErrAction::Read, ".");
        if !expect_action(e.file_type(), ErrAction::Read, ".").is_dir() {
            continue;
        }
        let Ok(mut name) = e.file_name().into_string() else {
            continue;
        };
        if name.starts_with("clippy_lints") && name != "clippy_lints_internal" {
            name.push_str("/src");
            for (file, module) in read_src_with_module(name.as_ref()) {
                parse_clippy_lint_decls(
                    file.path(),
                    File::open_read_to_cleared_string(file.path(), &mut contents),
                    &module,
                    &mut lints,
                );
            }
        }
    }
    lints.sort_by(|lhs, rhs| lhs.name.cmp(&rhs.name));
    lints
}

/// Reads the source files from the given root directory
fn read_src_with_module(src_root: &Path) -> impl use<'_> + Iterator<Item = (DirEntry, String)> {
    WalkDir::new(src_root).into_iter().filter_map(move |e| {
        let e = expect_action(e, ErrAction::Read, src_root);
        let path = e.path().as_os_str().as_encoded_bytes();
        if let Some(path) = path.strip_suffix(b".rs")
            && let Some(path) = path.get(src_root.as_os_str().len() + 1..)
        {
            if path == b"lib" {
                Some((e, String::new()))
            } else {
                let path = if let Some(path) = path.strip_suffix(b"mod")
                    && let Some(path) = path.strip_suffix(b"/").or_else(|| path.strip_suffix(b"\\"))
                {
                    path
                } else {
                    path
                };
                if let Ok(path) = str::from_utf8(path) {
                    let path = path.replace(['/', '\\'], "::");
                    Some((e, path))
                } else {
                    None
                }
            }
        } else {
            None
        }
    })
}

/// Parse a source file looking for `declare_clippy_lint` macro invocations.
fn parse_clippy_lint_decls(path: &Path, contents: &str, module: &str, lints: &mut Vec<Lint>) {
    #[allow(clippy::enum_glob_use)]
    use cursor::Pat::*;
    #[rustfmt::skip]
    static DECL_TOKENS: &[cursor::Pat<'_>] = &[
        // !{ /// docs
        Bang, OpenBrace, AnyComment,
        // #[clippy::version = "version"]
        Pound, OpenBracket, Ident("clippy"), DoubleColon, Ident("version"), Eq, LitStr, CloseBracket,
        // pub NAME, GROUP,
        Ident("pub"), CaptureIdent, Comma, AnyComment, CaptureIdent, Comma,
    ];

    let mut cursor = Cursor::new(contents);
    let mut captures = [Capture::EMPTY; 2];
    while let Some(start) = cursor.find_ident("declare_clippy_lint") {
        if cursor.match_all(DECL_TOKENS, &mut captures) && cursor.find_pat(CloseBrace) {
            lints.push(Lint {
                name: cursor.get_text(captures[0]).to_lowercase(),
                group: cursor.get_text(captures[1]).into(),
                module: module.into(),
                path: path.into(),
                declaration_range: start as usize..cursor.pos() as usize,
            });
        }
    }
}

#[must_use]
pub fn read_deprecated_lints() -> (Vec<DeprecatedLint>, Vec<RenamedLint>) {
    #[allow(clippy::enum_glob_use)]
    use cursor::Pat::*;
    #[rustfmt::skip]
    static DECL_TOKENS: &[cursor::Pat<'_>] = &[
        // #[clippy::version = "version"]
        Pound, OpenBracket, Ident("clippy"), DoubleColon, Ident("version"), Eq, CaptureLitStr, CloseBracket,
        // ("first", "second"),
        OpenParen, CaptureLitStr, Comma, CaptureLitStr, CloseParen, Comma,
    ];
    #[rustfmt::skip]
    static DEPRECATED_TOKENS: &[cursor::Pat<'_>] = &[
        // !{ DEPRECATED(DEPRECATED_VERSION) = [
        Bang, OpenBrace, Ident("DEPRECATED"), OpenParen, Ident("DEPRECATED_VERSION"), CloseParen, Eq, OpenBracket,
    ];
    #[rustfmt::skip]
    static RENAMED_TOKENS: &[cursor::Pat<'_>] = &[
        // !{ RENAMED(RENAMED_VERSION) = [
        Bang, OpenBrace, Ident("RENAMED"), OpenParen, Ident("RENAMED_VERSION"), CloseParen, Eq, OpenBracket,
    ];

    let path = "clippy_lints/src/deprecated_lints.rs";
    let mut deprecated = Vec::with_capacity(30);
    let mut renamed = Vec::with_capacity(80);
    let mut contents = String::new();
    File::open_read_to_cleared_string(path, &mut contents);

    let mut cursor = Cursor::new(&contents);
    let mut captures = [Capture::EMPTY; 3];

    // First instance is the macro definition.
    assert!(
        cursor.find_ident("declare_with_version").is_some(),
        "error reading deprecated lints"
    );

    if cursor.find_ident("declare_with_version").is_some() && cursor.match_all(DEPRECATED_TOKENS, &mut []) {
        while cursor.match_all(DECL_TOKENS, &mut captures) {
            deprecated.push(DeprecatedLint {
                name: parse_str_single_line(path.as_ref(), cursor.get_text(captures[1])),
                reason: parse_str_single_line(path.as_ref(), cursor.get_text(captures[2])),
                version: parse_str_single_line(path.as_ref(), cursor.get_text(captures[0])),
            });
        }
    } else {
        panic!("error reading deprecated lints");
    }

    if cursor.find_ident("declare_with_version").is_some() && cursor.match_all(RENAMED_TOKENS, &mut []) {
        while cursor.match_all(DECL_TOKENS, &mut captures) {
            renamed.push(RenamedLint {
                old_name: parse_str_single_line(path.as_ref(), cursor.get_text(captures[1])),
                new_name: parse_str_single_line(path.as_ref(), cursor.get_text(captures[2])),
                version: parse_str_single_line(path.as_ref(), cursor.get_text(captures[0])),
            });
        }
    } else {
        panic!("error reading renamed lints");
    }

    deprecated.sort_by(|lhs, rhs| lhs.name.cmp(&rhs.name));
    renamed.sort_by(|lhs, rhs| lhs.old_name.cmp(&rhs.old_name));
    (deprecated, renamed)
}

/// Removes the line splices and surrounding quotes from a string literal
fn parse_str_lit(s: &str) -> String {
    let (s, is_raw) = if let Some(s) = s.strip_prefix("r") {
        (s.trim_matches('#'), true)
    } else {
        (s, false)
    };
    let s = s
        .strip_prefix('"')
        .and_then(|s| s.strip_suffix('"'))
        .unwrap_or_else(|| panic!("expected quoted string, found `{s}`"));

    if is_raw {
        s.into()
    } else {
        let mut res = String::with_capacity(s.len());
        rustc_literal_escaper::unescape_str(s, &mut |_, ch| {
            if let Ok(ch) = ch {
                res.push(ch);
            }
        });
        res
    }
}

fn parse_str_single_line(path: &Path, s: &str) -> String {
    let value = parse_str_lit(s);
    assert!(
        !value.contains('\n'),
        "error parsing `{}`: `{s}` should be a single line string",
        path.display(),
    );
    value
}
