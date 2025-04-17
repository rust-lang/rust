use core::fmt::{self, Display};
use core::ops::Range;
use core::slice;
use core::str::FromStr;
use rustc_lexer::{self as lexer, FrontmatterAllowed};
use std::env;
use std::ffi::OsStr;
use std::fs::{self, OpenOptions};
use std::io::{self, Read as _, Seek as _, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::{self, Command, ExitStatus, Stdio};

#[cfg(not(windows))]
static CARGO_CLIPPY_EXE: &str = "cargo-clippy";
#[cfg(windows)]
static CARGO_CLIPPY_EXE: &str = "cargo-clippy.exe";

#[derive(Clone, Copy)]
pub enum ErrAction {
    Open,
    Read,
    Write,
    Create,
    Rename,
    Delete,
    Run,
}
impl ErrAction {
    fn as_str(self) -> &'static str {
        match self {
            Self::Open => "opening",
            Self::Read => "reading",
            Self::Write => "writing",
            Self::Create => "creating",
            Self::Rename => "renaming",
            Self::Delete => "deleting",
            Self::Run => "running",
        }
    }
}

#[cold]
#[track_caller]
pub fn panic_action(err: &impl Display, action: ErrAction, path: &Path) -> ! {
    panic!("error {} `{}`: {}", action.as_str(), path.display(), *err)
}

/// Wrapper around `std::fs::File` which panics with a path on failure.
pub struct File<'a> {
    pub inner: fs::File,
    pub path: &'a Path,
}
impl<'a> File<'a> {
    /// Opens a file panicking on failure.
    #[track_caller]
    pub fn open(path: &'a (impl AsRef<Path> + ?Sized), options: &mut OpenOptions) -> Self {
        let path = path.as_ref();
        match options.open(path) {
            Ok(inner) => Self { inner, path },
            Err(e) => panic_action(&e, ErrAction::Open, path),
        }
    }

    /// Opens a file if it exists, panicking on any other failure.
    #[track_caller]
    pub fn open_if_exists(path: &'a (impl AsRef<Path> + ?Sized), options: &mut OpenOptions) -> Option<Self> {
        let path = path.as_ref();
        match options.open(path) {
            Ok(inner) => Some(Self { inner, path }),
            Err(e) if e.kind() == io::ErrorKind::NotFound => None,
            Err(e) => panic_action(&e, ErrAction::Open, path),
        }
    }

    /// Opens and reads a file into a string, panicking of failure.
    #[track_caller]
    pub fn open_read_to_cleared_string<'dst>(
        path: &'a (impl AsRef<Path> + ?Sized),
        dst: &'dst mut String,
    ) -> &'dst mut String {
        Self::open(path, OpenOptions::new().read(true)).read_to_cleared_string(dst)
    }

    /// Read the entire contents of a file to the given buffer.
    #[track_caller]
    pub fn read_append_to_string<'dst>(&mut self, dst: &'dst mut String) -> &'dst mut String {
        match self.inner.read_to_string(dst) {
            Ok(_) => {},
            Err(e) => panic_action(&e, ErrAction::Read, self.path),
        }
        dst
    }

    #[track_caller]
    pub fn read_to_cleared_string<'dst>(&mut self, dst: &'dst mut String) -> &'dst mut String {
        dst.clear();
        self.read_append_to_string(dst)
    }

    /// Replaces the entire contents of a file.
    #[track_caller]
    pub fn replace_contents(&mut self, data: &[u8]) {
        let res = match self.inner.seek(SeekFrom::Start(0)) {
            Ok(_) => match self.inner.write_all(data) {
                Ok(()) => self.inner.set_len(data.len() as u64),
                Err(e) => Err(e),
            },
            Err(e) => Err(e),
        };
        if let Err(e) = res {
            panic_action(&e, ErrAction::Write, self.path);
        }
    }
}

/// Returns the path to the `cargo-clippy` binary
///
/// # Panics
///
/// Panics if the path of current executable could not be retrieved.
#[must_use]
pub fn cargo_clippy_path() -> PathBuf {
    let mut path = env::current_exe().expect("failed to get current executable name");
    path.set_file_name(CARGO_CLIPPY_EXE);
    path
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Version {
    pub major: u16,
    pub minor: u16,
}
impl FromStr for Version {
    type Err = ();
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if let Some(s) = s.strip_prefix("0.")
            && let Some((major, minor)) = s.split_once('.')
            && let Ok(major) = major.parse()
            && let Ok(minor) = minor.parse()
        {
            Ok(Self { major, minor })
        } else {
            Err(())
        }
    }
}
impl Version {
    /// Displays the version as a rust version. i.e. `x.y.0`
    #[must_use]
    pub fn rust_display(self) -> impl Display {
        struct X(Version);
        impl Display for X {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}.{}.0", self.0.major, self.0.minor)
            }
        }
        X(self)
    }

    /// Displays the version as it should appear in clippy's toml files. i.e. `0.x.y`
    #[must_use]
    pub fn toml_display(self) -> impl Display {
        struct X(Version);
        impl Display for X {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "0.{}.{}", self.0.major, self.0.minor)
            }
        }
        X(self)
    }
}

enum TomlPart<'a> {
    Table(&'a str),
    Value(&'a str, &'a str),
}

fn toml_iter(s: &str) -> impl Iterator<Item = (usize, TomlPart<'_>)> {
    let mut pos = 0;
    s.split('\n')
        .map(move |s| {
            let x = pos;
            pos += s.len() + 1;
            (x, s)
        })
        .filter_map(|(pos, s)| {
            if let Some(s) = s.strip_prefix('[') {
                s.split_once(']').map(|(name, _)| (pos, TomlPart::Table(name)))
            } else if matches!(s.bytes().next(), Some(b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'_')) {
                s.split_once('=').map(|(key, value)| (pos, TomlPart::Value(key, value)))
            } else {
                None
            }
        })
}

pub struct CargoPackage<'a> {
    pub name: &'a str,
    pub version_range: Range<usize>,
    pub not_a_platform_range: Range<usize>,
}

#[must_use]
pub fn parse_cargo_package(s: &str) -> CargoPackage<'_> {
    let mut in_package = false;
    let mut in_platform_deps = false;
    let mut name = "";
    let mut version_range = 0..0;
    let mut not_a_platform_range = 0..0;
    for (offset, part) in toml_iter(s) {
        match part {
            TomlPart::Table(name) => {
                if in_platform_deps {
                    not_a_platform_range.end = offset;
                }
                in_package = false;
                in_platform_deps = false;

                match name.trim() {
                    "package" => in_package = true,
                    "target.'cfg(NOT_A_PLATFORM)'.dependencies" => {
                        in_platform_deps = true;
                        not_a_platform_range.start = offset;
                    },
                    _ => {},
                }
            },
            TomlPart::Value(key, value) if in_package => match key.trim_end() {
                "name" => name = value.trim(),
                "version" => {
                    version_range.start = offset + (value.len() - value.trim().len()) + key.len() + 1;
                    version_range.end = offset + key.len() + value.trim_end().len() + 1;
                },
                _ => {},
            },
            TomlPart::Value(..) => {},
        }
    }
    CargoPackage {
        name,
        version_range,
        not_a_platform_range,
    }
}

pub struct ClippyInfo {
    pub path: PathBuf,
    pub version: Version,
    pub has_intellij_hook: bool,
}
impl ClippyInfo {
    #[must_use]
    pub fn search_for_manifest() -> Self {
        let mut path = env::current_dir().expect("error reading the working directory");
        let mut buf = String::new();
        loop {
            path.push("Cargo.toml");
            if let Some(mut file) = File::open_if_exists(&path, OpenOptions::new().read(true)) {
                file.read_to_cleared_string(&mut buf);
                let package = parse_cargo_package(&buf);
                if package.name == "\"clippy\"" {
                    if let Some(version) = buf[package.version_range].strip_prefix('"')
                        && let Some(version) = version.strip_suffix('"')
                        && let Ok(version) = version.parse()
                    {
                        path.pop();
                        return ClippyInfo {
                            path,
                            version,
                            has_intellij_hook: !package.not_a_platform_range.is_empty(),
                        };
                    }
                    panic!("error reading clippy version from `{}`", file.path.display());
                }
            }

            path.pop();
            assert!(
                path.pop(),
                "error finding project root, please run from inside the clippy directory"
            );
        }
    }
}

/// # Panics
/// Panics if given command result was failed.
pub fn exit_if_err(status: io::Result<ExitStatus>) {
    match status.expect("failed to run command").code() {
        Some(0) => {},
        Some(n) => process::exit(n),
        None => {
            eprintln!("Killed by signal");
            process::exit(1);
        },
    }
}

#[derive(Clone, Copy)]
pub enum UpdateStatus {
    Unchanged,
    Changed,
}
impl UpdateStatus {
    #[must_use]
    pub fn from_changed(value: bool) -> Self {
        if value { Self::Changed } else { Self::Unchanged }
    }

    #[must_use]
    pub fn is_changed(self) -> bool {
        matches!(self, Self::Changed)
    }
}

#[derive(Clone, Copy)]
pub enum UpdateMode {
    Change,
    Check,
}
impl UpdateMode {
    #[must_use]
    pub fn from_check(check: bool) -> Self {
        if check { Self::Check } else { Self::Change }
    }

    #[must_use]
    pub fn is_check(self) -> bool {
        matches!(self, Self::Check)
    }
}

#[derive(Default)]
pub struct FileUpdater {
    src_buf: String,
    dst_buf: String,
}
impl FileUpdater {
    fn update_file_checked_inner(
        &mut self,
        tool: &str,
        mode: UpdateMode,
        path: &Path,
        update: &mut dyn FnMut(&Path, &str, &mut String) -> UpdateStatus,
    ) {
        let mut file = File::open(path, OpenOptions::new().read(true).write(true));
        file.read_to_cleared_string(&mut self.src_buf);
        self.dst_buf.clear();
        match (mode, update(path, &self.src_buf, &mut self.dst_buf)) {
            (UpdateMode::Check, UpdateStatus::Changed) => {
                eprintln!(
                    "the contents of `{}` are out of date\nplease run `{tool}` to update",
                    path.display()
                );
                process::exit(1);
            },
            (UpdateMode::Change, UpdateStatus::Changed) => file.replace_contents(self.dst_buf.as_bytes()),
            (UpdateMode::Check | UpdateMode::Change, UpdateStatus::Unchanged) => {},
        }
    }

    fn update_file_inner(&mut self, path: &Path, update: &mut dyn FnMut(&Path, &str, &mut String) -> UpdateStatus) {
        let mut file = File::open(path, OpenOptions::new().read(true).write(true));
        file.read_to_cleared_string(&mut self.src_buf);
        self.dst_buf.clear();
        if update(path, &self.src_buf, &mut self.dst_buf).is_changed() {
            file.replace_contents(self.dst_buf.as_bytes());
        }
    }

    pub fn update_file_checked(
        &mut self,
        tool: &str,
        mode: UpdateMode,
        path: impl AsRef<Path>,
        update: &mut dyn FnMut(&Path, &str, &mut String) -> UpdateStatus,
    ) {
        self.update_file_checked_inner(tool, mode, path.as_ref(), update);
    }

    #[expect(clippy::type_complexity)]
    pub fn update_files_checked(
        &mut self,
        tool: &str,
        mode: UpdateMode,
        files: &mut [(
            impl AsRef<Path>,
            &mut dyn FnMut(&Path, &str, &mut String) -> UpdateStatus,
        )],
    ) {
        for (path, update) in files {
            self.update_file_checked_inner(tool, mode, path.as_ref(), update);
        }
    }

    pub fn update_file(
        &mut self,
        path: impl AsRef<Path>,
        update: &mut dyn FnMut(&Path, &str, &mut String) -> UpdateStatus,
    ) {
        self.update_file_inner(path.as_ref(), update);
    }
}

/// Replaces a region in a text delimited by two strings. Returns the new text if both delimiters
/// were found, or the missing delimiter if not.
pub fn update_text_region(
    path: &Path,
    start: &str,
    end: &str,
    src: &str,
    dst: &mut String,
    insert: &mut impl FnMut(&mut String),
) -> UpdateStatus {
    let Some((src_start, src_end)) = src.split_once(start) else {
        panic!("`{}` does not contain `{start}`", path.display());
    };
    let Some((replaced_text, src_end)) = src_end.split_once(end) else {
        panic!("`{}` does not contain `{end}`", path.display());
    };
    dst.push_str(src_start);
    dst.push_str(start);
    let new_start = dst.len();
    insert(dst);
    let changed = dst[new_start..] != *replaced_text;
    dst.push_str(end);
    dst.push_str(src_end);
    UpdateStatus::from_changed(changed)
}

pub fn update_text_region_fn(
    start: &str,
    end: &str,
    mut insert: impl FnMut(&mut String),
) -> impl FnMut(&Path, &str, &mut String) -> UpdateStatus {
    move |path, src, dst| update_text_region(path, start, end, src, dst, &mut insert)
}

#[derive(Clone, Copy)]
pub enum Token<'a> {
    /// Matches any number of comments / doc comments.
    AnyComment,
    Ident(&'a str),
    CaptureIdent,
    LitStr,
    CaptureLitStr,
    Bang,
    CloseBrace,
    CloseBracket,
    CloseParen,
    /// This will consume the first colon even if the second doesn't exist.
    DoubleColon,
    Comma,
    Eq,
    Lifetime,
    Lt,
    Gt,
    OpenBrace,
    OpenBracket,
    OpenParen,
    Pound,
    Semi,
    Slash,
}

pub struct RustSearcher<'txt> {
    text: &'txt str,
    cursor: lexer::Cursor<'txt>,
    pos: u32,
    next_token: lexer::Token,
}
impl<'txt> RustSearcher<'txt> {
    #[must_use]
    #[expect(clippy::inconsistent_struct_constructor)]
    pub fn new(text: &'txt str) -> Self {
        let mut cursor = lexer::Cursor::new(text, FrontmatterAllowed::Yes);
        Self {
            text,
            pos: 0,
            next_token: cursor.advance_token(),
            cursor,
        }
    }

    #[must_use]
    pub fn peek_text(&self) -> &'txt str {
        &self.text[self.pos as usize..(self.pos + self.next_token.len) as usize]
    }

    #[must_use]
    pub fn peek_len(&self) -> u32 {
        self.next_token.len
    }

    #[must_use]
    pub fn peek(&self) -> lexer::TokenKind {
        self.next_token.kind
    }

    #[must_use]
    pub fn pos(&self) -> u32 {
        self.pos
    }

    #[must_use]
    pub fn at_end(&self) -> bool {
        self.next_token.kind == lexer::TokenKind::Eof
    }

    pub fn step(&mut self) {
        // `next_len` is zero for the sentinel value and the eof marker.
        self.pos += self.next_token.len;
        self.next_token = self.cursor.advance_token();
    }

    /// Consumes the next token if it matches the requested value and captures the value if
    /// requested. Returns true if a token was matched.
    fn read_token(&mut self, token: Token<'_>, captures: &mut slice::IterMut<'_, &mut &'txt str>) -> bool {
        loop {
            match (token, self.next_token.kind) {
                (_, lexer::TokenKind::Whitespace)
                | (
                    Token::AnyComment,
                    lexer::TokenKind::BlockComment { terminated: true, .. } | lexer::TokenKind::LineComment { .. },
                ) => self.step(),
                (Token::AnyComment, _) => return true,
                (Token::Bang, lexer::TokenKind::Bang)
                | (Token::CloseBrace, lexer::TokenKind::CloseBrace)
                | (Token::CloseBracket, lexer::TokenKind::CloseBracket)
                | (Token::CloseParen, lexer::TokenKind::CloseParen)
                | (Token::Comma, lexer::TokenKind::Comma)
                | (Token::Eq, lexer::TokenKind::Eq)
                | (Token::Lifetime, lexer::TokenKind::Lifetime { .. })
                | (Token::Lt, lexer::TokenKind::Lt)
                | (Token::Gt, lexer::TokenKind::Gt)
                | (Token::OpenBrace, lexer::TokenKind::OpenBrace)
                | (Token::OpenBracket, lexer::TokenKind::OpenBracket)
                | (Token::OpenParen, lexer::TokenKind::OpenParen)
                | (Token::Pound, lexer::TokenKind::Pound)
                | (Token::Semi, lexer::TokenKind::Semi)
                | (Token::Slash, lexer::TokenKind::Slash)
                | (
                    Token::LitStr,
                    lexer::TokenKind::Literal {
                        kind: lexer::LiteralKind::Str { terminated: true } | lexer::LiteralKind::RawStr { .. },
                        ..
                    },
                ) => {
                    self.step();
                    return true;
                },
                (Token::Ident(x), lexer::TokenKind::Ident) if x == self.peek_text() => {
                    self.step();
                    return true;
                },
                (Token::DoubleColon, lexer::TokenKind::Colon) => {
                    self.step();
                    if !self.at_end() && matches!(self.next_token.kind, lexer::TokenKind::Colon) {
                        self.step();
                        return true;
                    }
                    return false;
                },
                (
                    Token::CaptureLitStr,
                    lexer::TokenKind::Literal {
                        kind: lexer::LiteralKind::Str { terminated: true } | lexer::LiteralKind::RawStr { .. },
                        ..
                    },
                )
                | (Token::CaptureIdent, lexer::TokenKind::Ident) => {
                    **captures.next().unwrap() = self.peek_text();
                    self.step();
                    return true;
                },
                _ => return false,
            }
        }
    }

    #[must_use]
    pub fn find_token(&mut self, token: Token<'_>) -> bool {
        let mut capture = [].iter_mut();
        while !self.read_token(token, &mut capture) {
            self.step();
            if self.at_end() {
                return false;
            }
        }
        true
    }

    #[must_use]
    pub fn find_capture_token(&mut self, token: Token<'_>) -> Option<&'txt str> {
        let mut res = "";
        let mut capture = &mut res;
        let mut capture = slice::from_mut(&mut capture).iter_mut();
        while !self.read_token(token, &mut capture) {
            self.step();
            if self.at_end() {
                return None;
            }
        }
        Some(res)
    }

    #[must_use]
    pub fn match_tokens(&mut self, tokens: &[Token<'_>], captures: &mut [&mut &'txt str]) -> bool {
        let mut captures = captures.iter_mut();
        tokens.iter().all(|&t| self.read_token(t, &mut captures))
    }
}

#[expect(clippy::must_use_candidate)]
pub fn try_rename_file(old_name: &Path, new_name: &Path) -> bool {
    match OpenOptions::new().create_new(true).write(true).open(new_name) {
        Ok(file) => drop(file),
        Err(e) if matches!(e.kind(), io::ErrorKind::AlreadyExists | io::ErrorKind::NotFound) => return false,
        Err(ref e) => panic_action(e, ErrAction::Create, new_name),
    }
    match fs::rename(old_name, new_name) {
        Ok(()) => true,
        Err(ref e) => {
            drop(fs::remove_file(new_name));
            // `NotADirectory` happens on posix when renaming a directory to an existing file.
            // Windows will ignore this and rename anyways.
            if matches!(e.kind(), io::ErrorKind::NotFound | io::ErrorKind::NotADirectory) {
                false
            } else {
                panic_action(e, ErrAction::Rename, old_name);
            }
        },
    }
}

#[expect(clippy::must_use_candidate)]
pub fn try_rename_dir(old_name: &Path, new_name: &Path) -> bool {
    match fs::create_dir(new_name) {
        Ok(()) => {},
        Err(e) if matches!(e.kind(), io::ErrorKind::AlreadyExists | io::ErrorKind::NotFound) => return false,
        Err(ref e) => panic_action(e, ErrAction::Create, new_name),
    }
    // Windows can't reliably rename to an empty directory.
    #[cfg(windows)]
    drop(fs::remove_dir(new_name));
    match fs::rename(old_name, new_name) {
        Ok(()) => true,
        Err(ref e) => {
            // Already dropped earlier on windows.
            #[cfg(not(windows))]
            drop(fs::remove_dir(new_name));
            // `NotADirectory` happens on posix when renaming a file to an existing directory.
            if matches!(e.kind(), io::ErrorKind::NotFound | io::ErrorKind::NotADirectory) {
                false
            } else {
                panic_action(e, ErrAction::Rename, old_name);
            }
        },
    }
}

pub fn write_file(path: &Path, contents: &str) {
    fs::write(path, contents).unwrap_or_else(|e| panic_action(&e, ErrAction::Write, path));
}

#[must_use]
pub fn run_with_output(path: &(impl AsRef<Path> + ?Sized), cmd: &mut Command) -> Vec<u8> {
    fn f(path: &Path, cmd: &mut Command) -> Vec<u8> {
        match cmd
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .output()
        {
            Ok(x) => match x.status.exit_ok() {
                Ok(()) => x.stdout,
                Err(ref e) => panic_action(e, ErrAction::Run, path),
            },
            Err(ref e) => panic_action(e, ErrAction::Run, path),
        }
    }
    f(path.as_ref(), cmd)
}

pub fn run_with_args_split(
    mut make_cmd: impl FnMut() -> Command,
    mut run_cmd: impl FnMut(&mut Command),
    args: impl Iterator<Item: AsRef<OsStr>>,
) {
    let mut cmd = make_cmd();
    let mut len = 0;
    for arg in args {
        len += arg.as_ref().len();
        cmd.arg(arg);
        // Very conservative limit
        if len > 10000 {
            run_cmd(&mut cmd);
            cmd = make_cmd();
            len = 0;
        }
    }
    if len != 0 {
        run_cmd(&mut cmd);
    }
}

#[expect(clippy::must_use_candidate)]
pub fn delete_file_if_exists(path: &Path) -> bool {
    match fs::remove_file(path) {
        Ok(()) => true,
        Err(e) if matches!(e.kind(), io::ErrorKind::NotFound | io::ErrorKind::IsADirectory) => false,
        Err(ref e) => panic_action(e, ErrAction::Delete, path),
    }
}

pub fn delete_dir_if_exists(path: &Path) {
    match fs::remove_dir_all(path) {
        Ok(()) => {},
        Err(e) if matches!(e.kind(), io::ErrorKind::NotFound | io::ErrorKind::NotADirectory) => {},
        Err(ref e) => panic_action(e, ErrAction::Delete, path),
    }
}
