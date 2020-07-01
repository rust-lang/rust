//! Snapshot testing library, see
//! https://github.com/rust-analyzer/rust-analyzer/pull/5101
use std::{
    collections::HashMap,
    env, fmt, fs, mem,
    ops::Range,
    panic,
    path::{Path, PathBuf},
    sync::Mutex,
};

use difference::Changeset;
use once_cell::sync::Lazy;
use stdx::{lines_with_ends, trim_indent};

const HELP: &str = "
You can update all `expect![[]]` tests by running:

    env UPDATE_EXPECT=1 cargo test

To update a single test, place the cursor on `expect` token and use `run` feature of rust-analyzer.
";

fn update_expect() -> bool {
    env::var("UPDATE_EXPECT").is_ok()
}

/// expect![[r#"inline snapshot"#]]
#[macro_export]
macro_rules! expect {
    [[$data:literal]] => {$crate::Expect {
        position: $crate::Position {
            file: file!(),
            line: line!(),
            column: column!(),
        },
        data: $data,
    }};
    [[]] => { $crate::expect![[""]] };
}

/// expect_file!["/crates/foo/test_data/bar.html"]
#[macro_export]
macro_rules! expect_file {
    [$path:literal] => {$crate::ExpectFile { path: $path }};
}

#[derive(Debug)]
pub struct Expect {
    pub position: Position,
    pub data: &'static str,
}

#[derive(Debug)]
pub struct ExpectFile {
    pub path: &'static str,
}

#[derive(Debug)]
pub struct Position {
    pub file: &'static str,
    pub line: u32,
    pub column: u32,
}

impl fmt::Display for Position {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}:{}", self.file, self.line, self.column)
    }
}

impl Expect {
    pub fn assert_eq(&self, actual: &str) {
        let trimmed = self.trimmed();
        if &trimmed == actual {
            return;
        }
        Runtime::fail_expect(self, &trimmed, actual);
    }
    pub fn assert_debug_eq(&self, actual: &impl fmt::Debug) {
        let actual = format!("{:#?}\n", actual);
        self.assert_eq(&actual)
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
                let pat = "expect![[";
                let offset = line.find(pat).unwrap();
                let literal_start = line_start + offset + pat.len();
                let indent = line.chars().take_while(|&it| it == ' ').count();
                target_line = Some((literal_start, indent));
                break;
            }
            line_start += line.len();
        }
        let (literal_start, line_indent) = target_line.unwrap();
        let literal_length =
            file[literal_start..].find("]]").expect("Couldn't find matching `]]` for `expect![[`.");
        let literal_range = literal_start..literal_start + literal_length;
        Location { line_indent, literal_range }
    }
}

impl ExpectFile {
    pub fn assert_eq(&self, actual: &str) {
        let expected = self.read();
        if actual == expected {
            return;
        }
        Runtime::fail_file(self, &expected, actual);
    }
    fn read(&self) -> String {
        fs::read_to_string(self.abs_path()).unwrap_or_default().replace("\r\n", "\n")
    }
    fn write(&self, contents: &str) {
        fs::write(self.abs_path(), contents).unwrap()
    }
    fn abs_path(&self) -> PathBuf {
        workspace_root().join(self.path)
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
            println!("\x1b[1m\x1b[92mupdating\x1b[0m: {}", expect.path);
            expect.write(actual);
            return;
        }
        rt.panic(expect.path.to_string(), expected, actual);
    }

    fn panic(&mut self, position: String, expected: &str, actual: &str) {
        let print_help = !mem::replace(&mut self.help_printed, true);
        let help = if print_help { HELP } else { "" };

        let diff = Changeset::new(actual, expected, "\n");

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
            position, help, expected, actual, diff
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
        let path = workspace_root().join(expect.position.file);
        let original_text = fs::read_to_string(&path).unwrap();
        let patchwork = Patchwork::new(original_text.clone());
        FileRuntime { path, original_text, patchwork }
    }
    fn update(&mut self, expect: &Expect, actual: &str) {
        let loc = expect.locate(&self.original_text);
        let patch = format_patch(loc.line_indent.clone(), actual);
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

fn format_patch(line_indent: usize, patch: &str) -> String {
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
    let indent = &" ".repeat(line_indent);
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
        if is_multiline {
            buf.push_str(indent);
            buf.push_str("    ");
        }
        buf.push_str(line);
        final_newline = line.ends_with('\n');
    }
    if final_newline {
        buf.push_str(indent);
    }
    buf.push('"');
    buf.push_str(hashes);
    buf
}

fn workspace_root() -> PathBuf {
    Path::new(
        &env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| env!("CARGO_MANIFEST_DIR").to_owned()),
    )
    .ancestors()
    .nth(2)
    .unwrap()
    .to_path_buf()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_patch() {
        let patch = format_patch(0, "hello\nworld\n");
        expect![[r##"
            r#"
                hello
                world
            "#"##]]
        .assert_eq(&patch);

        let patch = format_patch(4, "single line");
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
}
