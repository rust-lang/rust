//! Minimalistic snapshot testing for Rust.
//!
//! # Introduction
//!
//! `expect_test` is a small addition over plain `assert_eq!` testing approach,
//! which allows to automatically update tests results.
//!
//! The core of the library is the `expect!` macro. It can be though of as a
//! super-charged string literal, which can update itself.
//!
//! Let's see an example:
//!
//! ```no_run
//! use expect_test::expect;
//!
//! let actual = 2 + 2;
//! let expected = expect![["5"]];
//! expected.assert_eq(&actual.to_string())
//! ```
//!
//! Running this code will produce a test failure, as `"5"` is indeed not equal
//! to `"4"`. Running the test with `UPDATE_EXPECT=1` env variable however would
//! "magically" update the code to:
//!
//! ```no_run
//! # use expect_test::expect;
//! let actual = 2 + 2;
//! let expected = expect![["4"]];
//! expected.assert_eq(&actual.to_string())
//! ```
//!
//! This becomes very useful when you have a lot of tests with verbose and
//! potentially changing expected output.
//!
//! Under the hood, the `expect!` macro uses `file!`, `line!` and `column!` to
//! record source position at compile time. At runtime, this position is used
//! to patch the file in-place, if `UPDATE_EXPECT` is set.
//!
//! # Guide
//!
//! `expect!` returns an instance of `Expect` struct, which holds position
//! information and a string literal. Use `Expect::assert_eq` for string
//! comparison. Use `Expect::assert_debug_eq` for verbose debug comparison. Note
//! that leading indentation is automatically removed.
//!
//! ```
//! use expect_test::expect;
//!
//! #[derive(Debug)]
//! struct Foo {
//!     value: i32,
//! }
//!
//! let actual = Foo { value: 92 };
//! let expected = expect![["
//!     Foo {
//!         value: 92,
//!     }
//! "]];
//! expected.assert_debug_eq(&actual);
//! ```
//!
//! Be careful with `assert_debug_eq` - in general, stability of the debug
//! representation is not guaranteed. However, even if it changes, you can
//! quickly update all the tests by running the test suite with `UPDATE_EXPECT`
//! environmental variable set.
//!
//! If the expected data is too verbose to include inline, you can store it in
//! an external file using the `expect_file!` macro:
//!
//! ```no_run
//! use expect_test::expect_file;
//!
//! let actual = 42;
//! let expected = expect_file!["./the-answer.txt"];
//! expected.assert_eq(&actual.to_string());
//! ```
//!
//! File path is relative to the current file.
//!
//! # Suggested Workflows
//!
//! I like to use data-driven tests with `expect_test`. I usually define a
//! single driver function `check` and then call it from individual tests:
//!
//! ```
//! use expect_test::{expect, Expect};
//!
//! fn check(actual: i32, expect: Expect) {
//!     let actual = actual.to_string();
//!     expect.assert_eq(&actual);
//! }
//!
//! #[test]
//! fn test_addition() {
//!     check(90 + 2, expect![["92"]]);
//! }
//!
//! #[test]
//! fn test_multiplication() {
//!     check(46 * 2, expect![["92"]]);
//! }
//! ```
//!
//! Each test's body is a single call to `check`. All the variation in tests
//! comes from the input data.
//!
//! When writing a new test, I usually copy-paste an old one, leave the `expect`
//! blank and use `UPDATE_EXPECT` to fill the value for me:
//!
//! ```
//! # use expect_test::{expect, Expect};
//! # fn check(_: i32, _: Expect) {}
//! #[test]
//! fn test_division() {
//!     check(92 / 2, expect![[]])
//! }
//! ```
//!
//! See
//! https://blog.janestreet.com/using-ascii-waveforms-to-test-hardware-designs/
//! for a cool example of snapshot testing in the wild!
//!
//! # Alternatives
//!
//! * [insta](https://crates.io/crates/insta) - a more feature full snapshot
//!   testing library.
//! * [k9](https://crates.io/crates/k9) - a testing library which includes
//!   support for snapshot testing among other things.
//!
//! # Maintenance status
//!
//! The main customer of this library is rust-analyzer. The library is  stable,
//! it is planned to not release any major versions past 1.0.
//!
//! ## Minimal Supported Rust Version
//!
//! This crate's minimum supported `rustc` version is `1.45.0`. MSRV is updated
//! conservatively, supporting roughly 10 minor versions of `rustc`. MSRV bump
//! is not considered semver breaking, but will require at least minor version
//! bump.
use std::{
    collections::HashMap,
    convert::TryInto,
    env, fmt, fs, mem,
    ops::Range,
    panic,
    path::{Path, PathBuf},
    sync::Mutex,
};

use once_cell::sync::{Lazy, OnceCell};

const HELP: &str = "
You can update all `expect![[]]` tests by running:

    env UPDATE_EXPECT=1 cargo test

To update a single test, place the cursor on `expect` token and use `run` feature of rust-analyzer.
";

fn update_expect() -> bool {
    env::var("UPDATE_EXPECT").is_ok()
}

/// Creates an instance of `Expect` from string literal:
///
/// ```
/// # use expect_test::expect;
/// expect![["
///     Foo { value: 92 }
/// "]];
/// ```
///
/// Leading indentation is stripped.
#[macro_export]
macro_rules! expect {
    [[$data:literal]] => {$crate::Expect {
        position: $crate::Position {
            file: file!(),
            line: line!(),
            column: column!(),
        },
        data: $data,
        indent: true,
    }};
    [[]] => { $crate::expect![[""]] };
}

/// Creates an instance of `ExpectFile` from relative or absolute path:
///
/// ```
/// # use expect_test::expect_file;
/// expect_file!["./test_data/bar.html"];
/// ```
#[macro_export]
macro_rules! expect_file {
    [$path:expr] => {$crate::ExpectFile {
        path: std::path::PathBuf::from($path),
        position: file!(),
    }};
}

/// Self-updating string literal.
#[derive(Debug)]
pub struct Expect {
    #[doc(hidden)]
    pub position: Position,
    #[doc(hidden)]
    pub data: &'static str,
    #[doc(hidden)]
    pub indent: bool,
}

/// Self-updating file.
#[derive(Debug)]
pub struct ExpectFile {
    #[doc(hidden)]
    pub path: PathBuf,
    #[doc(hidden)]
    pub position: &'static str,
}

/// Position of original `expect!` in the source file.
#[derive(Debug)]
pub struct Position {
    #[doc(hidden)]
    pub file: &'static str,
    #[doc(hidden)]
    pub line: u32,
    #[doc(hidden)]
    pub column: u32,
}

impl fmt::Display for Position {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}:{}", self.file, self.line, self.column)
    }
}

impl Expect {
    /// Checks if this expect is equal to `actual`.
    pub fn assert_eq(&self, actual: &str) {
        let trimmed = self.trimmed();
        if trimmed == actual {
            return;
        }
        Runtime::fail_expect(self, &trimmed, actual);
    }
    /// Checks if this expect is equal to `format!("{:#?}", actual)`.
    pub fn assert_debug_eq(&self, actual: &impl fmt::Debug) {
        let actual = format!("{:#?}\n", actual);
        self.assert_eq(&actual)
    }
    /// If `true` (default), in-place update will indent the string literal.
    pub fn indent(&mut self, yes: bool) {
        self.indent = yes;
    }

    fn trimmed(&self) -> String {
        if !self.data.contains('\n') {
            return self.data.to_string();
        }
        trim_indent(self.data)
    }

    fn locate(&self, file: &str) -> Location {
        let mut target_line = None;
        let mut line_start = 0;
        for (i, line) in lines_with_ends(file).enumerate() {
            if i == self.position.line as usize - 1 {
                // `column` points to the first character of the macro invocation:
                //
                //    expect![[ ... ]]
                //    ^
                //
                // Seek past the exclam, then skip any whitespace and
                // delimiters to get to our argument.
                let byte_offset = line
                    .char_indices()
                    .skip((self.position.column - 1).try_into().unwrap())
                    .skip_while(|&(_, c)| c != '!')
                    .skip(1)
                    .skip_while(|&(_, c)| c.is_whitespace())
                    .skip_while(|&(_, c)| matches!(c, '[' | '(' | '{'))
                    .next()
                    .expect("Failed to parse macro invocation")
                    .0;

                let literal_start = line_start + byte_offset;
                let indent = line.chars().take_while(|&it| it == ' ').count();
                target_line = Some((literal_start, indent));
                break;
            }
            line_start += line.len();
        }
        let (literal_start, line_indent) = target_line.unwrap();

        let lit_to_eof = &file[literal_start..];
        let lit_to_eof_trimmed = lit_to_eof.trim_start();

        let literal_start = literal_start + (lit_to_eof.len() - lit_to_eof_trimmed.len());

        let literal_len =
            locate_end(lit_to_eof_trimmed).expect("Couldn't find matching `]]` for `expect![[`.");
        let literal_range = literal_start..literal_start + literal_len;
        Location { line_indent, literal_range }
    }
}

fn locate_end(lit_to_eof: &str) -> Option<usize> {
    assert!(lit_to_eof.chars().next().map_or(true, |c| !c.is_whitespace()));

    if lit_to_eof.starts_with("]]") {
        // expect![[ ]]
        Some(0)
    } else {
        // expect![["foo"]]
        find_str_lit_len(lit_to_eof)
    }
}

/// Parses a string literal, returning the byte index of its last character
/// (either a quote or a hash).
fn find_str_lit_len(str_lit_to_eof: &str) -> Option<usize> {
    use StrLitKind::*;
    #[derive(Clone, Copy)]
    enum StrLitKind {
        Normal,
        Raw(usize),
    }

    fn try_find_n_hashes(
        s: &mut impl Iterator<Item = char>,
        desired_hashes: usize,
    ) -> Option<(usize, Option<char>)> {
        let mut n = 0;
        loop {
            match s.next()? {
                '#' => n += 1,
                c => return Some((n, Some(c))),
            }

            if n == desired_hashes {
                return Some((n, None));
            }
        }
    }

    let mut s = str_lit_to_eof.chars();
    let kind = match s.next()? {
        '"' => Normal,
        'r' => {
            let (n, c) = try_find_n_hashes(&mut s, usize::MAX)?;
            if c != Some('"') {
                return None;
            }
            Raw(n)
        }
        _ => return None,
    };

    let mut oldc = None;
    loop {
        let c = oldc.take().or_else(|| s.next())?;
        match (c, kind) {
            ('\\', Normal) => {
                let _escaped = s.next()?;
            }
            ('"', Normal) => break,
            ('"', Raw(0)) => break,
            ('"', Raw(n)) => {
                let (seen, c) = try_find_n_hashes(&mut s, n)?;
                if seen == n {
                    break;
                }
                oldc = c;
            }
            _ => {}
        }
    }

    Some(str_lit_to_eof.len() - s.as_str().len())
}

impl ExpectFile {
    /// Checks if file contents is equal to `actual`.
    pub fn assert_eq(&self, actual: &str) {
        let expected = self.read();
        if actual == expected {
            return;
        }
        Runtime::fail_file(self, &expected, actual);
    }
    /// Checks if file contents is equal to `format!("{:#?}", actual)`.
    pub fn assert_debug_eq(&self, actual: &impl fmt::Debug) {
        let actual = format!("{:#?}\n", actual);
        self.assert_eq(&actual)
    }
    fn read(&self) -> String {
        fs::read_to_string(self.abs_path()).unwrap_or_default().replace("\r\n", "\n")
    }
    fn write(&self, contents: &str) {
        fs::write(self.abs_path(), contents).unwrap()
    }
    fn abs_path(&self) -> PathBuf {
        if self.path.is_absolute() {
            self.path.to_owned()
        } else {
            let dir = Path::new(self.position).parent().unwrap();
            to_abs_ws_path(&dir.join(&self.path))
        }
    }
}

#[derive(Default)]
struct Runtime {
    help_printed: bool,
    per_file: HashMap<&'static str, FileRuntime>,
}
static RT: Lazy<Mutex<Runtime>> = Lazy::new(Default::default);

impl Runtime {
    fn fail_expect(expect: &Expect, expected: &str, actual: &str) {
        let mut rt = RT.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        if update_expect() {
            println!("\x1b[1m\x1b[92mupdating\x1b[0m: {}", expect.position);
            rt.per_file
                .entry(expect.position.file)
                .or_insert_with(|| FileRuntime::new(expect))
                .update(expect, actual);
            return;
        }
        rt.panic(expect.position.to_string(), expected, actual);
    }
    fn fail_file(expect: &ExpectFile, expected: &str, actual: &str) {
        let mut rt = RT.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        if update_expect() {
            println!("\x1b[1m\x1b[92mupdating\x1b[0m: {}", expect.path.display());
            expect.write(actual);
            return;
        }
        rt.panic(expect.path.display().to_string(), expected, actual);
    }
    fn panic(&mut self, position: String, expected: &str, actual: &str) {
        let print_help = !mem::replace(&mut self.help_printed, true);
        let help = if print_help { HELP } else { "" };

        let diff = dissimilar::diff(expected, actual);

        println!(
            "\n
\x1b[1m\x1b[91merror\x1b[97m: expect test failed\x1b[0m
   \x1b[1m\x1b[34m-->\x1b[0m {}
{}
\x1b[1mExpect\x1b[0m:
----
{}
----

\x1b[1mActual\x1b[0m:
----
{}
----

\x1b[1mDiff\x1b[0m:
----
{}
----
",
            position,
            help,
            expected,
            actual,
            format_chunks(diff)
        );
        // Use resume_unwind instead of panic!() to prevent a backtrace, which is unnecessary noise.
        panic::resume_unwind(Box::new(()));
    }
}

struct FileRuntime {
    path: PathBuf,
    original_text: String,
    patchwork: Patchwork,
}

impl FileRuntime {
    fn new(expect: &Expect) -> FileRuntime {
        let path = to_abs_ws_path(Path::new(expect.position.file));
        let original_text = fs::read_to_string(&path).unwrap();
        let patchwork = Patchwork::new(original_text.clone());
        FileRuntime { path, original_text, patchwork }
    }
    fn update(&mut self, expect: &Expect, actual: &str) {
        let loc = expect.locate(&self.original_text);
        let desired_indent = if expect.indent { Some(loc.line_indent) } else { None };
        let patch = format_patch(desired_indent, actual);
        self.patchwork.patch(loc.literal_range, &patch);
        fs::write(&self.path, &self.patchwork.text).unwrap()
    }
}

#[derive(Debug)]
struct Location {
    line_indent: usize,
    literal_range: Range<usize>,
}

#[derive(Debug)]
struct Patchwork {
    text: String,
    indels: Vec<(Range<usize>, usize)>,
}

impl Patchwork {
    fn new(text: String) -> Patchwork {
        Patchwork { text, indels: Vec::new() }
    }
    fn patch(&mut self, mut range: Range<usize>, patch: &str) {
        self.indels.push((range.clone(), patch.len()));
        self.indels.sort_by_key(|(delete, _insert)| delete.start);

        let (delete, insert) = self
            .indels
            .iter()
            .take_while(|(delete, _)| delete.start < range.start)
            .map(|(delete, insert)| (delete.end - delete.start, insert))
            .fold((0usize, 0usize), |(x1, y1), (x2, y2)| (x1 + x2, y1 + y2));

        for pos in &mut [&mut range.start, &mut range.end] {
            **pos -= delete;
            **pos += insert;
        }

        self.text.replace_range(range, &patch);
    }
}

fn format_patch(desired_indent: Option<usize>, patch: &str) -> String {
    let mut max_hashes = 0;
    let mut cur_hashes = 0;
    for byte in patch.bytes() {
        if byte != b'#' {
            cur_hashes = 0;
            continue;
        }
        cur_hashes += 1;
        max_hashes = max_hashes.max(cur_hashes);
    }
    let hashes = &"#".repeat(max_hashes + 1);
    let indent = desired_indent.map(|it| " ".repeat(it));
    let is_multiline = patch.contains('\n');

    let mut buf = String::new();
    buf.push('r');
    buf.push_str(hashes);
    buf.push('"');
    if is_multiline {
        buf.push('\n');
    }
    let mut final_newline = false;
    for line in lines_with_ends(patch) {
        if is_multiline && !line.trim().is_empty() {
            if let Some(indent) = &indent {
                buf.push_str(indent);
                buf.push_str("    ");
            }
        }
        buf.push_str(line);
        final_newline = line.ends_with('\n');
    }
    if final_newline {
        if let Some(indent) = &indent {
            buf.push_str(indent);
        }
    }
    buf.push('"');
    buf.push_str(hashes);
    buf
}

fn to_abs_ws_path(path: &Path) -> PathBuf {
    if path.is_absolute() {
        return path.to_owned();
    }

    static WORKSPACE_ROOT: OnceCell<PathBuf> = OnceCell::new();
    WORKSPACE_ROOT
        .get_or_try_init(|| {
            let my_manifest = env::var("CARGO_MANIFEST_DIR")?;

            // Heuristic, see https://github.com/rust-lang/cargo/issues/3946
            let workspace_root = Path::new(&my_manifest)
                .ancestors()
                .filter(|it| it.join("Cargo.toml").exists())
                .last()
                .unwrap()
                .to_path_buf();

            Ok(workspace_root)
        })
        .unwrap_or_else(|_: env::VarError| {
            panic!("No CARGO_MANIFEST_DIR env var and the path is relative: {}", path.display())
        })
        .join(path)
}

fn trim_indent(mut text: &str) -> String {
    if text.starts_with('\n') {
        text = &text[1..];
    }
    let indent = text
        .lines()
        .filter(|it| !it.trim().is_empty())
        .map(|it| it.len() - it.trim_start().len())
        .min()
        .unwrap_or(0);

    lines_with_ends(text)
        .map(
            |line| {
                if line.len() <= indent {
                    line.trim_start_matches(' ')
                } else {
                    &line[indent..]
                }
            },
        )
        .collect()
}

fn lines_with_ends(text: &str) -> LinesWithEnds {
    LinesWithEnds { text }
}

struct LinesWithEnds<'a> {
    text: &'a str,
}

impl<'a> Iterator for LinesWithEnds<'a> {
    type Item = &'a str;
    fn next(&mut self) -> Option<&'a str> {
        if self.text.is_empty() {
            return None;
        }
        let idx = self.text.find('\n').map_or(self.text.len(), |it| it + 1);
        let (res, next) = self.text.split_at(idx);
        self.text = next;
        Some(res)
    }
}

fn format_chunks(chunks: Vec<dissimilar::Chunk>) -> String {
    let mut buf = String::new();
    for chunk in chunks {
        let formatted = match chunk {
            dissimilar::Chunk::Equal(text) => text.into(),
            dissimilar::Chunk::Delete(text) => format!("\x1b[41m{}\x1b[0m", text),
            dissimilar::Chunk::Insert(text) => format!("\x1b[42m{}\x1b[0m", text),
        };
        buf.push_str(&formatted);
    }
    buf
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_patch() {
        let patch = format_patch(None, "hello\nworld\n");
        expect![[r##"
            r#"
            hello
            world
            "#"##]]
        .assert_eq(&patch);

        let patch = format_patch(Some(0), "hello\nworld\n");
        expect![[r##"
            r#"
                hello
                world
            "#"##]]
        .assert_eq(&patch);

        let patch = format_patch(Some(4), "single line");
        expect![[r##"r#"single line"#"##]].assert_eq(&patch);
    }

    #[test]
    fn test_patchwork() {
        let mut patchwork = Patchwork::new("one two three".to_string());
        patchwork.patch(4..7, "zwei");
        patchwork.patch(0..3, "один");
        patchwork.patch(8..13, "3");
        expect![[r#"
            Patchwork {
                text: "один zwei 3",
                indels: [
                    (
                        0..3,
                        8,
                    ),
                    (
                        4..7,
                        4,
                    ),
                    (
                        8..13,
                        1,
                    ),
                ],
            }
        "#]]
        .assert_debug_eq(&patchwork);
    }

    #[test]
    fn test_expect_file() {
        expect_file!["./lib.rs"].assert_eq(include_str!("./lib.rs"))
    }

    #[test]
    fn smoke_test_indent() {
        fn check_indented(input: &str, mut expect: Expect) {
            expect.indent(true);
            expect.assert_eq(input);
        }
        fn check_not_indented(input: &str, mut expect: Expect) {
            expect.indent(false);
            expect.assert_eq(input);
        }

        check_indented(
            "\
line1
  line2
",
            expect![[r#"
                line1
                  line2
            "#]],
        );

        check_not_indented(
            "\
line1
  line2
",
            expect![[r#"
line1
  line2
"#]],
        );
    }

    #[test]
    fn test_locate() {
        macro_rules! check_locate {
            ($( [[$s:literal]] ),* $(,)?) => {$({
                let lit = stringify!($s);
                let with_trailer = format!("{} \t]]\n", lit);
                assert_eq!(locate_end(&with_trailer), Some(lit.len()));
            })*};
        }

        // Check that we handle string literals containing "]]" correctly.
        check_locate!(
            [[r#"{ arr: [[1, 2], [3, 4]], other: "foo" } "#]],
            [["]]"]],
            [["\"]]"]],
            [[r#""]]"#]],
        );

        // Check `expect![[  ]]` as well.
        assert_eq!(locate_end("]]"), Some(0));
    }

    #[test]
    fn test_find_str_lit_len() {
        macro_rules! check_str_lit_len {
            ($( $s:literal ),* $(,)?) => {$({
                let lit = stringify!($s);
                assert_eq!(find_str_lit_len(lit), Some(lit.len()));
            })*}
        }

        check_str_lit_len![
            r##"foa\""#"##,
            r##"

                asdf][]]""""#
            "##,
            "",
            "\"",
            "\"\"",
            "#\"#\"#",
        ];
    }
}
