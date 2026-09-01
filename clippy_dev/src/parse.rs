pub mod cursor;

use self::cursor::{Capture, Cursor, IdentPat};
use crate::utils::{ErrAction, Scoped, StrBuf, VecBuf, expect_action, slice_groups_mut, walk_dir_no_dot_or_target};
use crate::{DiagCx, SourceFile, Span};
use core::fmt::{self, Display};
use core::range::Range;
use rustc_arena::{DroplessArena, TypedArena};
use rustc_data_structures::fx::FxHashMap;
use std::collections::hash_map::{Entry, VacantEntry};
use std::{fs, path};

pub struct ParseCxImpl<'cx> {
    pub arena: &'cx DroplessArena,
    pub source_files: &'cx TypedArena<SourceFile<'cx>>,
    pub str_buf: StrBuf,
    pub str_list_buf: VecBuf<&'cx str>,
    pub dcx: DiagCx,
}
pub type ParseCx<'cx> = &'cx mut ParseCxImpl<'cx>;

/// Calls the given function inside a newly created parsing context.
pub fn new_parse_cx<'env, T>(f: impl for<'cx> FnOnce(&'cx mut Scoped<'cx, 'env, ParseCxImpl<'cx>>) -> T) -> T {
    let arena = DroplessArena::default();
    let source_files = TypedArena::default();
    f(&mut Scoped::new(ParseCxImpl {
        arena: &arena,
        source_files: &source_files,
        str_buf: StrBuf::with_capacity(128),
        str_list_buf: VecBuf::with_capacity(128),
        dcx: DiagCx::default(),
    }))
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum LintTool {
    Rustc,
    Clippy,
}
impl LintTool {
    /// Gets the namespace prefix to use when naming a lint including the `::`.
    pub fn prefix(self) -> &'static str {
        match self {
            Self::Rustc => "",
            Self::Clippy => "clippy::",
        }
    }

    pub fn from_prefix(s: &str) -> Option<Self> {
        (s == "clippy").then_some(Self::Clippy)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct LintName<'cx> {
    pub tool: LintTool,
    pub name: &'cx str,
}
impl<'cx> LintName<'cx> {
    pub fn new_rustc(name: &'cx str) -> Self {
        Self {
            tool: LintTool::Rustc,
            name,
        }
    }

    pub fn new_clippy(name: &'cx str) -> Self {
        Self {
            tool: LintTool::Clippy,
            name,
        }
    }
}
impl Display for LintName<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.tool.prefix())?;
        f.write_str(self.name)
    }
}

pub struct ActiveLint<'cx> {
    pub group: &'cx str,
    pub decl_range: Range<u32>,
}

pub struct DeprecatedLint<'cx> {
    pub reason: &'cx str,
    pub version: &'cx str,
}

pub struct RenamedLint<'cx> {
    pub new_name: LintName<'cx>,
    pub version: &'cx str,
}

pub enum LintData<'cx> {
    Active(ActiveLint<'cx>),
    Deprecated(DeprecatedLint<'cx>),
    Renamed(RenamedLint<'cx>),
}

pub struct Lint<'cx> {
    pub name_sp: Span<'cx>,
    pub data: LintData<'cx>,
}

#[derive(Clone, Copy)]
pub enum LintPassMac {
    Declare,
    Impl,
}
impl LintPassMac {
    pub fn name(self) -> &'static str {
        match self {
            Self::Declare => "declare_lint_pass",
            Self::Impl => "impl_lint_pass",
        }
    }
}

pub struct LintPass<'cx> {
    /// The raw text of the documentation comments. May include leading/trailing
    /// whitespace and empty lines.
    pub docs: &'cx str,
    pub name: &'cx str,
    pub lt: Option<&'cx str>,
    pub mac: LintPassMac,
    pub decl_sp: Span<'cx>,
    pub lints: &'cx mut [&'cx str],
}

pub struct ParsedLints<'cx> {
    pub lints: FxHashMap<&'cx str, Lint<'cx>>,
    pub lint_passes: Vec<LintPass<'cx>>,
    pub deprecated_file: &'cx SourceFile<'cx>,
}
impl<'cx> ParsedLints<'cx> {
    #[expect(clippy::mutable_key_type)]
    pub fn mk_file_to_lint_decl_map(&self) -> FxHashMap<&'cx SourceFile<'cx>, Vec<(&'cx str, Range<u32>)>> {
        #[expect(clippy::default_trait_access)]
        let mut lints = FxHashMap::with_capacity_and_hasher(500, Default::default());
        for (&name, lint) in &self.lints {
            if let LintData::Active(lint_data) = &lint.data {
                lints
                    .entry(lint.name_sp.file)
                    .or_insert_with(|| Vec::with_capacity(8))
                    .push((name, lint_data.decl_range));
            }
        }
        lints
    }

    pub fn iter_passes_by_file_mut<'s>(&'s mut self) -> impl Iterator<Item = &'s mut [LintPass<'cx>]> {
        slice_groups_mut(&mut self.lint_passes, |head, tail| {
            tail.iter().take_while(|&x| x.decl_sp.file == head.decl_sp.file).count()
        })
    }

    #[track_caller]
    fn get_vacant_lint<'a>(
        &'a mut self,
        dcx: &mut DiagCx,
        name: &'cx str,
        name_sp: Span<'cx>,
    ) -> Option<VacantEntry<'a, &'cx str, Lint<'cx>>> {
        match self.lints.entry(name) {
            Entry::Vacant(e) => Some(e),
            Entry::Occupied(e) => {
                dcx.emit_duplicate_lint(name_sp, e.get().name_sp);
                None
            },
        }
    }
}

impl<'cx> ParseCxImpl<'cx> {
    /// Finds and parses all lint declarations.
    pub fn parse_lint_decls(&mut self) -> ParsedLints<'cx> {
        let mut data = ParsedLints {
            #[expect(clippy::default_trait_access)]
            lints: FxHashMap::with_capacity_and_hasher(1000, Default::default()),
            lint_passes: Vec::with_capacity(400),
            deprecated_file: self.source_files.alloc(SourceFile::load(self.str_buf.alloc_collect(
                self.arena,
                [
                    "clippy_lints",
                    path::MAIN_SEPARATOR_STR,
                    "src",
                    path::MAIN_SEPARATOR_STR,
                    "deprecated_lints.rs",
                ],
            ))),
        };

        for e in expect_action(fs::read_dir("."), ErrAction::Read, ".") {
            let e = expect_action(e, ErrAction::Read, ".");

            // Skip if this isn't a lint crate's directory.
            let mut crate_path = if expect_action(e.file_type(), ErrAction::Read, ".").is_dir()
                && let Ok(crate_path) = e.file_name().into_string()
                && crate_path.starts_with("clippy_lints")
                && crate_path != "clippy_lints_internal"
            {
                crate_path
            } else {
                continue;
            };

            crate_path.push(path::MAIN_SEPARATOR);
            crate_path.push_str("src");
            for e in walk_dir_no_dot_or_target(&crate_path) {
                let e = expect_action(e, ErrAction::Read, &crate_path);
                if e.path().as_os_str().as_encoded_bytes().ends_with(b".rs")
                    && let Some(file_path) = e.path().to_str()
                    && file_path != data.deprecated_file.path.get()
                {
                    let file = self
                        .source_files
                        .alloc(SourceFile::load(self.arena.alloc_str(file_path)));
                    self.parse_lint_src_file(&mut data, file);
                }
            }
        }

        self.parse_deprecated_lints(&mut data);
        data
    }

    /// Parse a source file looking for `declare_clippy_lint` macro invocations.
    fn parse_lint_src_file(&mut self, data: &mut ParsedLints<'cx>, file: &'cx SourceFile<'cx>) {
        #[allow(clippy::enum_glob_use)]
        use cursor::Pat::*;

        let mut cursor = Cursor::new(&file.contents);
        let mut captures = [Capture::EMPTY; 3];
        while let Some(mac_name) = cursor.find_capture_ident() {
            if !cursor.eat_bang() {
                continue;
            }
            match cursor.get_text(mac_name) {
                "declare_clippy_lint" => {
                    #[rustfmt::skip]
                    static DECL_START: &[cursor::Pat] = &[
                        // { /// docs
                        OpenBrace, AnyComments,
                        // #[clippy::version = "version"]
                        Pound, OpenBracket, Ident(IdentPat::clippy), DoubleColon,
                        Ident(IdentPat::version), Eq, CaptureLitStr, CloseBracket,
                        // pub NAME, GROUP, "desc",
                        Ident(IdentPat::r#pub), CaptureIdent, Comma,
                        AnyComments, CaptureIdent, Comma, AnyComments, LitStr,
                    ];
                    #[rustfmt::skip]
                    static OPTION: &[cursor::Pat] = &[
                        // @option = value
                        AnyComments, At, AnyIdent, Eq, Lit,
                    ];

                    if let Err(expected) = cursor
                        .match_all(DECL_START, &mut captures)
                        .and_then(|()| {
                            (!cursor.eat_comma()).ok_or(()).or_else(|()| {
                                cursor.eat_list(|cursor| cursor.match_all(OPTION, &mut []).map(|()| true))
                            })
                        })
                        .and_then(|()| cursor.eat_close_brace().ok_or("`}`"))
                    {
                        cursor.emit_unexpected(&mut self.dcx, file, expected);
                    } else if let name = self.str_buf.alloc_ascii_lower(self.arena, cursor.get_text(captures[1]))
                        && let name_sp = captures[1].mk_sp(file)
                        && let Some(e) = data.get_vacant_lint(&mut self.dcx, name, name_sp)
                    {
                        let _ = self.parse_version(cursor.get_text(captures[0]), captures[0].mk_sp(file));
                        e.insert(Lint {
                            name_sp,
                            data: LintData::Active(ActiveLint {
                                group: cursor.get_text(captures[2]),
                                decl_range: mac_name.pos..cursor.pos(),
                            }),
                        });
                    }
                },
                mac @ ("declare_lint_pass" | "impl_lint_pass") => {
                    let mut has_lt = false;
                    let mut lints: &mut [_] = &mut [];
                    if let Err(expected) = cursor
                        .match_all(&[OpenParen, CaptureDocLines, CaptureIdent], &mut captures)
                        .and_then(|()| cursor.opt_match_all(&[Lt, CaptureLifetime, Gt], &mut captures[2..]))
                        .and_then(|res| {
                            has_lt = res;
                            cursor.match_all(&[FatArrow, OpenBracket], &mut [])
                        })
                        .and_then(|()| {
                            cursor.capture_list(&mut self.str_list_buf, self.arena, |cursor| {
                                cursor.capture_opt_path(&mut self.str_buf, self.arena)
                            })
                        })
                        .and_then(|res| {
                            lints = res;
                            cursor.match_all(&[CloseBracket, CloseParen, Semi], &mut [])
                        })
                    {
                        cursor.emit_unexpected(&mut self.dcx, file, expected);
                    } else {
                        data.lint_passes.push(LintPass {
                            docs: cursor.get_text(captures[0]),
                            name: cursor.get_text(captures[1]),
                            lt: has_lt.then(|| cursor.get_text(captures[2])),
                            mac: if matches!(mac, "declare_lint_pass") {
                                LintPassMac::Declare
                            } else {
                                LintPassMac::Impl
                            },
                            decl_sp: Span::new(file, mac_name.pos..cursor.pos()),
                            lints,
                        });
                    }
                },
                _ => {},
            }
        }
    }

    fn parse_deprecated_lints(&mut self, data: &mut ParsedLints<'cx>) {
        #[allow(clippy::enum_glob_use)]
        use cursor::Pat::*;

        #[rustfmt::skip]
        static DECL_TOKENS: &[cursor::Pat] = &[
            // #[clippy::version = "version"]
            Pound, OpenBracket, Ident(IdentPat::clippy), DoubleColon,
            Ident(IdentPat::version), Eq, CaptureLitStr, CloseBracket,
            // ("first", "second")
            OpenParen, CaptureLitStr, Comma, CaptureLitStr, CloseParen,
        ];
        #[rustfmt::skip]
        static DEPRECATED_TOKENS: &[cursor::Pat] = &[
            // !{ DEPRECATED(DEPRECATED_VERSION) = [
            Bang, OpenBrace, Ident(IdentPat::DEPRECATED), OpenParen,
            Ident(IdentPat::DEPRECATED_VERSION), CloseParen, Eq, OpenBracket,
        ];
        #[rustfmt::skip]
        static RENAMED_TOKENS: &[cursor::Pat] = &[
            // !{ RENAMED(RENAMED_VERSION) = [
            Bang, OpenBrace, Ident(IdentPat::RENAMED), OpenParen,
            Ident(IdentPat::RENAMED_VERSION), CloseParen, Eq, OpenBracket,
        ];

        let file = data.deprecated_file;
        let mut cursor = Cursor::new(&file.contents);
        let mut captures = [Capture::EMPTY; 3];

        if let Err(expected) = cursor
            .find_ident("declare_with_version")
            .ok_or("`declare_with_version`")
            .and_then(|()| {
                cursor
                    .find_ident("declare_with_version")
                    .ok_or("`declare_with_version`")
            })
            .and_then(|()| cursor.match_all(DEPRECATED_TOKENS, &mut []))
            .and_then(|()| {
                cursor.eat_list(|cursor| {
                    let parsed = cursor.opt_match_all(DECL_TOKENS, &mut captures)?;
                    if parsed
                        && let [version, name, reason] = captures
                        && let name_sp = name.mk_sp(file)
                        && let (Some(version), Some(name), Some(reason)) = (
                            self.parse_version(cursor.get_text(version), version.mk_sp(file)),
                            self.parse_clippy_lint_name(cursor.get_text(name), name_sp),
                            self.parse_str_lit(cursor.get_text(reason), reason.mk_sp(file)),
                        )
                        && let Some(e) = data.get_vacant_lint(&mut self.dcx, name, name_sp)
                    {
                        e.insert(Lint {
                            name_sp,
                            data: LintData::Deprecated(DeprecatedLint { reason, version }),
                        });
                    }
                    Ok(parsed)
                })
            })
            .and_then(|()| {
                cursor
                    .find_ident("declare_with_version")
                    .ok_or("`declare_with_version`")
            })
            .and_then(|()| cursor.match_all(RENAMED_TOKENS, &mut []))
            .and_then(|()| {
                cursor.eat_list(|cursor| {
                    let parsed = cursor.opt_match_all(DECL_TOKENS, &mut captures)?;
                    if parsed
                        && let [version, name, new_name] = captures
                        && let name_sp = name.mk_sp(file)
                        && let (Some(version), Some(name), Some(new_name)) = (
                            self.parse_version(cursor.get_text(version), version.mk_sp(file)),
                            self.parse_clippy_lint_name(cursor.get_text(name), name_sp),
                            self.parse_lint_name(cursor.get_text(new_name), new_name.mk_sp(file)),
                        )
                        && let Some(e) = data.get_vacant_lint(&mut self.dcx, name, name_sp)
                    {
                        e.insert(Lint {
                            name_sp,
                            data: LintData::Renamed(RenamedLint { new_name, version }),
                        });
                    }
                    Ok(parsed)
                })
            })
        {
            cursor.emit_unexpected(&mut self.dcx, file, expected);
        }
    }

    /// Removes the line splices and surrounding quotes from a string literal.
    fn parse_str_lit(&mut self, s: &'cx str, sp: Span<'cx>) -> Option<&'cx str> {
        let (s, is_raw, sp_base) = if let Some(trimmed) = s.strip_prefix("r") {
            let trimmed = trimmed.trim_start_matches('#');
            #[expect(clippy::cast_possible_truncation)]
            let sp_base = (s.len() - trimmed.len() + 1) as u32;
            (trimmed.trim_end_matches('#'), true, sp_base)
        } else {
            (s, false, 1)
        };
        let sp_base = sp.range.start + sp_base;
        let s = s
            .strip_prefix('"')
            .and_then(|s| s.strip_suffix('"'))
            .unwrap_or_else(|| panic!("expected quoted string, found `{s}`"));

        let mut is_ok = true;
        if is_raw {
            rustc_literal_escaper::check_raw_str(s, |range, c| {
                if c.is_err_and(|e| e.is_fatal()) {
                    #[expect(clippy::cast_possible_truncation)]
                    let range = range.start as u32 + sp_base..range.end as u32 + sp_base;
                    self.dcx
                        .emit_spanned_err(Span::new(sp.file, range), "invalid string escape sequence");
                    is_ok = false;
                }
            });
            is_ok.then_some(s)
        } else {
            self.str_buf.with(|buf| {
                rustc_literal_escaper::unescape_str(s, |range, c| match c {
                    Ok(c) => buf.push(c),
                    Err(e) if e.is_fatal() => {
                        #[expect(clippy::cast_possible_truncation)]
                        let range = range.start as u32 + sp_base..range.end as u32 + sp_base;
                        self.dcx
                            .emit_spanned_err(Span::new(sp.file, range), "invalid string escape sequence");
                        is_ok = false;
                    },
                    Err(_) => {},
                });
                is_ok.then(|| {
                    if buf == s {
                        s
                    } else if buf.is_empty() {
                        ""
                    } else {
                        self.arena.alloc_str(buf)
                    }
                })
            })
        }
    }

    #[track_caller]
    fn parse_lint_name(&mut self, s: &'cx str, sp: Span<'cx>) -> Option<LintName<'cx>> {
        let s = self.parse_str_lit(s, sp)?;
        let (tool, name) = match s.split_once("::") {
            Some((tool, name)) if let Some(tool) = LintTool::from_prefix(tool) => (tool, name),
            Some(..) => {
                self.dcx.emit_spanned_err(sp, "unknown lint tool");
                return None;
            },
            None => (LintTool::Rustc, s),
        };
        if name
            .bytes()
            .all(|c| matches!(c, b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'_'))
        {
            Some(LintName { tool, name })
        } else {
            self.dcx.emit_spanned_err(sp, "unparsable lint name");
            None
        }
    }

    #[track_caller]
    fn parse_clippy_lint_name(&mut self, s: &'cx str, sp: Span<'cx>) -> Option<&'cx str> {
        let name = self.parse_lint_name(s, sp)?;
        if name.tool == LintTool::Clippy {
            Some(name.name)
        } else {
            self.dcx.emit_not_clippy_lint_name(sp);
            None
        }
    }

    #[track_caller]
    fn parse_version(&mut self, s: &'cx str, sp: Span<'cx>) -> Option<&'cx str> {
        let s = self.parse_str_lit(s, sp)?;
        if s.bytes().all(|c| matches!(c, b'0'..=b'9' | b'.')) || matches!(s, "pre 1.29.0" | "CURRENT_RUSTC_VERSION") {
            Some(s)
        } else {
            self.dcx.emit_spanned_err(sp, "unparsable version number");
            None
        }
    }
}
