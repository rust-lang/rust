pub mod cursor;

use self::cursor::{Capture, Cursor};
use crate::utils::{ErrAction, File, Scoped, expect_action, slice_groups_mut, walk_dir_no_dot_or_target};
use core::cell::Cell;
use core::fmt::{self, Display, Write as _};
use core::hash::{Hash, Hasher};
use core::ptr;
use core::range::Range;
use rustc_arena::{DroplessArena, TypedArena};
use rustc_data_structures::fx::FxHashMap;
use std::str::pattern::Pattern;
use std::{fs, path};

#[derive(Eq)]
pub struct SourceFile<'cx> {
    // `cargo dev rename_lint` needs to be able to rename files.
    pub path: Cell<&'cx str>,
    pub contents: String,
}
impl<'cx> SourceFile<'cx> {
    pub fn load(path: &'cx str) -> Self {
        let mut contents = String::new();
        File::open_read(path).read_append_to_string(&mut contents);
        SourceFile {
            path: Cell::new(path),
            contents,
        }
    }

    /// Splits the file's path into the crate it's a part of and the module it implements.
    ///
    /// Only supports paths in the form `CRATE_NAME/src/PATH/TO/FILE.rs` using the current
    /// platform's path separator. The module path returned will use the current platform's
    /// path separator.
    pub fn path_as_krate_mod(&self) -> (&'cx str, &'cx str) {
        let path = self.path.get();
        let Some((krate, path)) = path.split_once(path::MAIN_SEPARATOR) else {
            return ("", "");
        };
        let module = if let Some(path) = path.strip_prefix("src")
            && let Some(path) = path.strip_prefix(path::MAIN_SEPARATOR)
            && let Some(path) = path.strip_suffix(".rs")
        {
            if path == "lib" {
                ""
            } else if let Some(path) = path.strip_suffix("mod")
                && let Some(path) = path.strip_suffix(path::MAIN_SEPARATOR)
            {
                path
            } else {
                path
            }
        } else {
            ""
        };
        (krate, module)
    }
}
impl PartialEq<SourceFile<'_>> for SourceFile<'_> {
    fn eq(&self, other: &SourceFile<'_>) -> bool {
        // We should only be creating one source file per path.
        ptr::addr_eq(self, other)
    }
}
impl Hash for SourceFile<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        ptr::hash(self, state);
    }
}

pub struct ParseCxImpl<'cx> {
    pub arena: &'cx DroplessArena,
    pub source_files: &'cx TypedArena<SourceFile<'cx>>,
    pub str_buf: StrBuf,
    pub str_list_buf: VecBuf<&'cx str>,
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

    /// Collects all elements into the buffer and allocates that onto the arena.
    pub fn alloc_collect<'cx, I>(&mut self, arena: &'cx DroplessArena, iter: I) -> &'cx str
    where
        I: IntoIterator,
        String: Extend<I::Item>,
    {
        self.0.clear();
        self.0.extend(iter);
        if self.0.is_empty() {
            ""
        } else {
            arena.alloc_str(&self.0)
        }
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

pub struct VecBuf<T>(Vec<T>);
impl<T> VecBuf<T> {
    /// Creates a new buffer with the specified initial capacity.
    pub fn with_capacity(cap: usize) -> Self {
        Self(Vec::with_capacity(cap))
    }

    /// Performs an operation with the freshly cleared buffer.
    pub fn with<R>(&mut self, f: impl FnOnce(&mut Vec<T>) -> R) -> R {
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
    pub file: &'cx SourceFile<'cx>,
    pub group: &'cx str,
    pub declaration_range: Range<u32>,
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
    pub file: &'cx SourceFile<'cx>,
    pub decl_range: Range<u32>,
    pub lints: &'cx mut [&'cx str],
}

pub struct LintData<'cx> {
    pub lints: FxHashMap<&'cx str, Lint<'cx>>,
    pub lint_passes: Vec<LintPass<'cx>>,
    pub deprecated_file: &'cx SourceFile<'cx>,
}
impl<'cx> LintData<'cx> {
    #[expect(clippy::mutable_key_type)]
    pub fn mk_file_to_lint_decl_map(&self) -> FxHashMap<&'cx SourceFile<'cx>, Vec<(&'cx str, Range<u32>)>> {
        #[expect(clippy::default_trait_access)]
        let mut lints = FxHashMap::with_capacity_and_hasher(500, Default::default());
        for (&name, lint) in &self.lints {
            if let Lint::Active(lint) = lint {
                lints
                    .entry(lint.file)
                    .or_insert_with(|| Vec::with_capacity(8))
                    .push((name, lint.declaration_range));
            }
        }
        lints
    }

    pub fn iter_passes_by_file_mut<'s>(&'s mut self) -> impl Iterator<Item = &'s mut [LintPass<'cx>]> {
        slice_groups_mut(&mut self.lint_passes, |head, tail| {
            tail.iter().take_while(|&x| x.file == head.file).count()
        })
    }
}

impl<'cx> ParseCxImpl<'cx> {
    /// Finds and parses all lint declarations.
    #[must_use]
    pub fn parse_lint_decls(&mut self) -> LintData<'cx> {
        let mut data = LintData {
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
    fn parse_lint_src_file(&mut self, data: &mut LintData<'cx>, file: &'cx SourceFile<'cx>) {
        #[allow(clippy::enum_glob_use)]
        use cursor::Pat::*;
        #[rustfmt::skip]
        static LINT_DECL_TOKENS: &[cursor::Pat<'_>] = &[
            // !{ /// docs
            Bang, OpenBrace, AnyComment,
            // #[clippy::version = "version"]
            Pound, OpenBracket, Ident("clippy"), DoubleColon, Ident("version"), Eq, LitStr, CloseBracket,
            // pub NAME, GROUP,
            Ident("pub"), CaptureIdent, Comma, AnyComment, CaptureIdent, Comma,
        ];
        #[rustfmt::skip]
        static PASS_DECL_TOKENS: &[cursor::Pat<'_>] = &[
            // !( NAME <'lt> => [
            Bang, OpenParen, CaptureDocLines, CaptureIdent, CaptureOptLifetimeArg, FatArrow, OpenBracket,
        ];

        let mut cursor = Cursor::new(&file.contents);
        let mut captures = [Capture::EMPTY; 3];
        while let Some(mac_name) = cursor.find_any_ident() {
            match cursor.get_text(mac_name) {
                "declare_clippy_lint"
                    if cursor.match_all(LINT_DECL_TOKENS, &mut captures) && cursor.find_pat(CloseBrace) =>
                {
                    assert!(
                        data.lints
                            .insert(
                                self.str_buf.alloc_ascii_lower(self.arena, cursor.get_text(captures[0])),
                                Lint::Active(ActiveLint {
                                    file,
                                    group: cursor.get_text(captures[1]),
                                    declaration_range: mac_name.pos..cursor.pos(),
                                }),
                            )
                            .is_none()
                    );
                },
                mac @ ("declare_lint_pass" | "impl_lint_pass") if cursor.match_all(PASS_DECL_TOKENS, &mut captures) => {
                    let mac = if matches!(mac, "declare_lint_pass") {
                        LintPassMac::Declare
                    } else {
                        LintPassMac::Impl
                    };
                    let docs = cursor.get_text(captures[0]);
                    let name = cursor.get_text(captures[1]);
                    let lt = cursor.get_text(captures[2]);
                    let lt = if lt.is_empty() { None } else { Some(lt) };

                    let lints = self.str_list_buf.with(|buf| {
                        // Parses a comma separated list of paths and converts each path
                        // to a string with whitespace removed.
                        while !cursor.match_pat(CloseBracket) {
                            buf.push(self.str_buf.with(|buf| {
                                if cursor.match_pat(DoubleColon) {
                                    buf.push_str("::");
                                }
                                let capture = cursor.capture_ident()?;
                                buf.push_str(cursor.get_text(capture));
                                while cursor.match_pat(DoubleColon) {
                                    buf.push_str("::");
                                    let capture = cursor.capture_ident()?;
                                    buf.push_str(cursor.get_text(capture));
                                }
                                Some(self.arena.alloc_str(buf))
                            })?);

                            if !cursor.match_pat(Comma) {
                                if !cursor.match_pat(CloseBracket) {
                                    return None;
                                }
                                break;
                            }
                        }

                        // The arena panics when allocating a size of zero.
                        Some(if buf.is_empty() {
                            &mut []
                        } else {
                            self.arena.alloc_slice(buf)
                        })
                    });

                    if let Some(lints) = lints
                        && cursor.match_all(&[CloseParen, Semi], &mut [])
                    {
                        data.lint_passes.push(LintPass {
                            docs,
                            name,
                            lt,
                            mac,
                            file,
                            decl_range: mac_name.pos..cursor.pos(),
                            lints,
                        });
                    }
                },
                _ => {},
            }
        }
    }

    fn parse_deprecated_lints(&mut self, data: &mut LintData<'cx>) {
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

        let file = data.deprecated_file;
        let mut cursor = Cursor::new(&file.contents);
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
                            self.parse_clippy_lint_name(file, cursor.get_text(captures[1])),
                            Lint::Deprecated(DeprecatedLint {
                                reason: self.parse_str_single_line(file, cursor.get_text(captures[2])),
                                version: self.parse_str_single_line(file, cursor.get_text(captures[0])),
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
                            self.parse_clippy_lint_name(file, cursor.get_text(captures[1])),
                            Lint::Renamed(RenamedLint {
                                new_name: self.parse_lint_name(file, cursor.get_text(captures[2])),
                                version: self.parse_str_single_line(file, cursor.get_text(captures[0])),
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
    fn parse_str_lit(&mut self, s: &'cx str) -> &'cx str {
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
            s
        } else {
            self.str_buf.with(|buf| {
                rustc_literal_escaper::unescape_str(s, &mut |_, ch| {
                    if let Ok(ch) = ch {
                        buf.push(ch);
                    }
                });
                if buf == s {
                    s
                } else if buf.is_empty() {
                    ""
                } else {
                    self.arena.alloc_str(buf)
                }
            })
        }
    }

    fn parse_str_single_line(&mut self, file: &SourceFile<'_>, s: &'cx str) -> &'cx str {
        let value = self.parse_str_lit(s);
        assert!(
            !value.contains('\n'),
            "error parsing `{}`: `{s}` should be a single line string",
            file.path.get(),
        );
        value
    }

    fn parse_clippy_lint_name(&mut self, file: &SourceFile<'_>, s: &'cx str) -> &'cx str {
        match self.parse_str_single_line(file, s).strip_prefix("clippy::") {
            Some(x) => x,
            None => panic!(
                "error parsing `{}`: `{s}` should be a string starting with `clippy::`",
                file.path.get(),
            ),
        }
    }

    fn parse_lint_name(&mut self, file: &SourceFile<'_>, s: &'cx str) -> LintName<'cx> {
        let s = self.parse_str_single_line(file, s);
        let (name, tool) = match s.strip_prefix("clippy::") {
            Some(s) => (s, LintTool::Clippy),
            None => (s, LintTool::Rustc),
        };
        LintName { name, tool }
    }
}
