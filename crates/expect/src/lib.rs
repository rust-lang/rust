//! Snapshot testing library, see
//! https://github.com/rust-analyzer/rust-analyzer/pull/5101
use std::{
    collections::HashMap,
    env, fmt, fs,
    ops::Range,
    path::{Path, PathBuf},
    sync::Mutex,
};

use once_cell::sync::Lazy;
use stdx::{lines_with_ends, trim_indent};

const HELP: &str = "
You can update all `expect![[]]` tests by:

    env UPDATE_EXPECT=1 cargo test

To update a single test, place the cursor on `expect` token and use `run` feature of rust-analyzer.
";

fn update_expect() -> bool {
    env::var("UPDATE_EXPECT").is_ok()
}

/// expect![[""]]
#[macro_export]
macro_rules! expect {
    [[$lit:literal]] => {$crate::Expect {
        file: file!(),
        line: line!(),
        column: column!(),
        data: $lit,
    }};
    [[]] => { $crate::expect![[""]] };
}

#[derive(Debug)]
pub struct Expect {
    pub file: &'static str,
    pub line: u32,
    pub column: u32,
    pub data: &'static str,
}

impl Expect {
    pub fn assert_eq(&self, actual: &str) {
        let trimmed = self.trimmed();
        if &trimmed == actual {
            return;
        }
        Runtime::fail(self, &trimmed, actual);
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
            if i == self.line as usize - 1 {
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

#[derive(Default)]
struct Runtime {
    help_printed: bool,
    per_file: HashMap<&'static str, FileRuntime>,
}
static RT: Lazy<Mutex<Runtime>> = Lazy::new(Default::default);

impl Runtime {
    fn fail(expect: &Expect, expected: &str, actual: &str) {
        let mut rt = RT.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut updated = "";
        if update_expect() {
            updated = " (updated)";
            rt.per_file
                .entry(expect.file)
                .or_insert_with(|| FileRuntime::new(expect))
                .update(expect, actual);
        }
        let print_help = !rt.help_printed && !update_expect();
        rt.help_printed = true;

        let help = if print_help { HELP } else { "" };
        println!(
            "\n
error: expect test failed{}
  --> {}:{}:{}
{}
Expect:
----
{}
----

Actual:
----
{}
----
",
            updated, expect.file, expect.line, expect.column, help, expected, actual
        );
        // Use resume_unwind instead of panic!() to prevent a backtrace, which is unnecessary noise.
        std::panic::resume_unwind(Box::new(()));
    }
}

struct FileRuntime {
    path: PathBuf,
    original_text: String,
    patchwork: Patchwork,
}

impl FileRuntime {
    fn new(expect: &Expect) -> FileRuntime {
        let path = workspace_root().join(expect.file);
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
            **pos += insert;
            **pos -= delete
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
