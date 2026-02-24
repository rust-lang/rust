pub mod cursor;

use self::cursor::{Capture, Cursor};
use crate::utils::{ErrAction, File, Scoped, expect_action, walk_dir_no_dot_or_target};
use core::fmt::{self, Display, Write as _};
use core::range::Range;
use rustc_arena::DroplessArena;
use rustc_data_structures::fx::FxHashMap;
use std::fs;
use std::path::{self, Path, PathBuf};
use std::str::pattern::Pattern;

pub struct ParseCxImpl<'cx> {
    pub arena: &'cx DroplessArena,
    pub str_buf: StrBuf,
}
pub type ParseCx<'cx> = &'cx mut ParseCxImpl<'cx>;

/// Calls the given function inside a newly created parsing context.
pub fn new_parse_cx<'env, T>(f: impl for<'cx> FnOnce(&'cx mut Scoped<'cx, 'env, ParseCxImpl<'cx>>) -> T) -> T {
    let arena = DroplessArena::default();
    f(&mut Scoped::new(ParseCxImpl {
        arena: &arena,
        str_buf: StrBuf::with_capacity(128),
    }))
}

/// A string used as a temporary buffer used to avoid allocating for short lived strings.
pub struct StrBuf(String);
impl StrBuf {
    /// Creates a new buffer with the specified initial capacity.
    pub fn with_capacity(cap: usize) -> Self {
        Self(String::with_capacity(cap))
    }

    /// Allocates the result of formatting the given value onto the arena.
    pub fn alloc_display<'cx>(&mut self, arena: &'cx DroplessArena, value: impl Display) -> &'cx str {
        self.0.clear();
        write!(self.0, "{value}").expect("`Display` impl returned an error");
        arena.alloc_str(&self.0)
    }

    /// Allocates the string onto the arena with all ascii characters converted to
    /// lowercase.
    pub fn alloc_ascii_lower<'cx>(&mut self, arena: &'cx DroplessArena, s: &str) -> &'cx str {
        self.0.clear();
        self.0.push_str(s);
        self.0.make_ascii_lowercase();
        arena.alloc_str(&self.0)
    }

    /// Allocates the result of replacing all instances the pattern with the given string
    /// onto the arena.
    pub fn alloc_replaced<'cx>(
        &mut self,
        arena: &'cx DroplessArena,
        s: &str,
        pat: impl Pattern,
        replacement: &str,
    ) -> &'cx str {
        let mut parts = s.split(pat);
        let Some(first) = parts.next() else {
            return "";
        };
        self.0.clear();
        self.0.push_str(first);
        for part in parts {
            self.0.push_str(replacement);
            self.0.push_str(part);
        }
        if self.0.is_empty() {
            ""
        } else {
            arena.alloc_str(&self.0)
        }
    }

    /// Performs an operation with the freshly cleared buffer.
    pub fn with<T>(&mut self, f: impl FnOnce(&mut String) -> T) -> T {
        self.0.clear();
        f(&mut self.0)
    }
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
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct LintName<'cx> {
    pub name: &'cx str,
    pub tool: LintTool,
}
impl<'cx> LintName<'cx> {
    pub fn new_rustc(name: &'cx str) -> Self {
        Self {
            name,
            tool: LintTool::Rustc,
        }
    }

    pub fn new_clippy(name: &'cx str) -> Self {
        Self {
            name,
            tool: LintTool::Clippy,
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
    pub module: &'cx str,
    pub path: PathBuf,
    pub declaration_range: Range<usize>,
}

pub struct DeprecatedLint<'cx> {
    pub reason: &'cx str,
    pub version: &'cx str,
}

pub struct RenamedLint<'cx> {
    pub new_name: LintName<'cx>,
    pub version: &'cx str,
}

pub enum Lint<'cx> {
    Active(ActiveLint<'cx>),
    Deprecated(DeprecatedLint<'cx>),
    Renamed(RenamedLint<'cx>),
}

pub struct LintData<'cx> {
    pub lints: FxHashMap<&'cx str, Lint<'cx>>,
}

impl<'cx> ParseCxImpl<'cx> {
    /// Finds and parses all lint declarations.
    #[must_use]
    pub fn parse_lint_decls(&mut self) -> LintData<'cx> {
        let mut data = LintData {
            #[expect(clippy::default_trait_access)]
            lints: FxHashMap::with_capacity_and_hasher(1000, Default::default()),
        };

        let mut contents = String::new();
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
                if let Some(path) = e.path().to_str()
                    && let Some(path) = path.strip_suffix(".rs")
                    && let Some(path) = path.get(crate_path.len() + 1..)
                {
                    let module = if path == "lib" {
                        ""
                    } else {
                        let path = path
                            .strip_suffix("mod")
                            .and_then(|x| x.strip_suffix(path::MAIN_SEPARATOR))
                            .unwrap_or(path);
                        self.str_buf
                            .alloc_replaced(self.arena, path, path::MAIN_SEPARATOR, "::")
                    };
                    self.parse_clippy_lint_decls(
                        e.path(),
                        File::open_read_to_cleared_string(e.path(), &mut contents),
                        module,
                        &mut data,
                    );
                }
            }
        }

        self.read_deprecated_lints(&mut data);
        data
    }

    /// Parse a source file looking for `declare_clippy_lint` macro invocations.
    fn parse_clippy_lint_decls(&mut self, path: &Path, contents: &str, module: &'cx str, data: &mut LintData<'cx>) {
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
                assert!(
                    data.lints
                        .insert(
                            self.str_buf.alloc_ascii_lower(self.arena, cursor.get_text(captures[0])),
                            Lint::Active(ActiveLint {
                                group: self.arena.alloc_str(cursor.get_text(captures[1])),
                                module,
                                path: path.into(),
                                declaration_range: start as usize..cursor.pos() as usize,
                            }),
                        )
                        .is_none()
                );
            }
        }
    }

    fn read_deprecated_lints(&mut self, data: &mut LintData<'cx>) {
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
                assert!(
                    data.lints
                        .insert(
                            self.parse_clippy_lint_name(path.as_ref(), cursor.get_text(captures[1])),
                            Lint::Deprecated(DeprecatedLint {
                                reason: self.parse_str_single_line(path.as_ref(), cursor.get_text(captures[2])),
                                version: self.parse_str_single_line(path.as_ref(), cursor.get_text(captures[0])),
                            }),
                        )
                        .is_none()
                );
            }
        } else {
            panic!("error reading deprecated lints");
        }

        if cursor.find_ident("declare_with_version").is_some() && cursor.match_all(RENAMED_TOKENS, &mut []) {
            while cursor.match_all(DECL_TOKENS, &mut captures) {
                assert!(
                    data.lints
                        .insert(
                            self.parse_clippy_lint_name(path.as_ref(), cursor.get_text(captures[1])),
                            Lint::Renamed(RenamedLint {
                                new_name: self.parse_lint_name(path.as_ref(), cursor.get_text(captures[2])),
                                version: self.parse_str_single_line(path.as_ref(), cursor.get_text(captures[0])),
                            }),
                        )
                        .is_none()
                );
            }
        } else {
            panic!("error reading renamed lints");
        }
    }

    /// Removes the line splices and surrounding quotes from a string literal
    fn parse_str_lit(&mut self, s: &str) -> &'cx str {
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
            if s.is_empty() { "" } else { self.arena.alloc_str(s) }
        } else {
            self.str_buf.with(|buf| {
                rustc_literal_escaper::unescape_str(s, &mut |_, ch| {
                    if let Ok(ch) = ch {
                        buf.push(ch);
                    }
                });
                if buf.is_empty() { "" } else { self.arena.alloc_str(buf) }
            })
        }
    }

    fn parse_str_single_line(&mut self, path: &Path, s: &str) -> &'cx str {
        let value = self.parse_str_lit(s);
        assert!(
            !value.contains('\n'),
            "error parsing `{}`: `{s}` should be a single line string",
            path.display(),
        );
        value
    }

    fn parse_clippy_lint_name(&mut self, path: &Path, s: &str) -> &'cx str {
        match self.parse_str_single_line(path, s).strip_prefix("clippy::") {
            Some(x) => x,
            None => panic!(
                "error parsing `{}`: `{s}` should be a string starting with `clippy::`",
                path.display()
            ),
        }
    }

    fn parse_lint_name(&mut self, path: &Path, s: &str) -> LintName<'cx> {
        let s = self.parse_str_single_line(path, s);
        let (name, tool) = match s.strip_prefix("clippy::") {
            Some(s) => (s, LintTool::Clippy),
            None => (s, LintTool::Rustc),
        };
        LintName { name, tool }
    }
}
