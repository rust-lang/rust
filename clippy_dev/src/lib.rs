#![feature(once_cell)]
#![cfg_attr(feature = "deny-warnings", deny(warnings))]
// warn on lints, that are included in `rust-lang/rust`s bootstrap
#![warn(rust_2018_idioms, unused_lifetimes)]

use itertools::Itertools;
use regex::Regex;
use std::collections::HashMap;
use std::ffi::OsStr;
use std::fs;
use std::lazy::SyncLazy;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

pub mod bless;
pub mod fmt;
pub mod new_lint;
pub mod serve;
pub mod setup;
pub mod update_lints;

static DEC_CLIPPY_LINT_RE: SyncLazy<Regex> = SyncLazy::new(|| {
    Regex::new(
        r#"(?x)
    declare_clippy_lint!\s*[\{(]
    (?:\s+///.*)*
    \s+pub\s+(?P<name>[A-Z_][A-Z_0-9]*)\s*,\s*
    (?P<cat>[a-z_]+)\s*,\s*
    "(?P<desc>(?:[^"\\]+|\\(?s).(?-s))*)"\s*[})]
"#,
    )
    .unwrap()
});

static DEC_DEPRECATED_LINT_RE: SyncLazy<Regex> = SyncLazy::new(|| {
    Regex::new(
        r#"(?x)
    declare_deprecated_lint!\s*[{(]\s*
    (?:\s+///.*)*
    \s+pub\s+(?P<name>[A-Z_][A-Z_0-9]*)\s*,\s*
    "(?P<desc>(?:[^"\\]+|\\(?s).(?-s))*)"\s*[})]
"#,
    )
    .unwrap()
});
static NL_ESCAPE_RE: SyncLazy<Regex> = SyncLazy::new(|| Regex::new(r#"\\\n\s*"#).unwrap());

pub static DOCS_LINK: &str = "https://rust-lang.github.io/rust-clippy/master/index.html";

/// Lint data parsed from the Clippy source code.
#[derive(Clone, PartialEq, Debug)]
pub struct Lint {
    pub name: String,
    pub group: String,
    pub desc: String,
    pub deprecation: Option<String>,
    pub module: String,
}

impl Lint {
    #[must_use]
    pub fn new(name: &str, group: &str, desc: &str, deprecation: Option<&str>, module: &str) -> Self {
        Self {
            name: name.to_lowercase(),
            group: group.to_string(),
            desc: NL_ESCAPE_RE.replace(&desc.replace("\\\"", "\""), "").to_string(),
            deprecation: deprecation.map(ToString::to_string),
            module: module.to_string(),
        }
    }

    /// Returns all non-deprecated lints and non-internal lints
    #[must_use]
    pub fn usable_lints(lints: &[Self]) -> Vec<Self> {
        lints
            .iter()
            .filter(|l| l.deprecation.is_none() && !l.group.starts_with("internal"))
            .cloned()
            .collect()
    }

    /// Returns all internal lints (not `internal_warn` lints)
    #[must_use]
    pub fn internal_lints(lints: &[Self]) -> Vec<Self> {
        lints.iter().filter(|l| l.group == "internal").cloned().collect()
    }

    /// Returns all deprecated lints
    #[must_use]
    pub fn deprecated_lints(lints: &[Self]) -> Vec<Self> {
        lints.iter().filter(|l| l.deprecation.is_some()).cloned().collect()
    }

    /// Returns the lints in a `HashMap`, grouped by the different lint groups
    #[must_use]
    pub fn by_lint_group(lints: impl Iterator<Item = Self>) -> HashMap<String, Vec<Self>> {
        lints.map(|lint| (lint.group.to_string(), lint)).into_group_map()
    }
}

/// Generates the Vec items for `register_lint_group` calls in `clippy_lints/src/lib.rs`.
#[must_use]
pub fn gen_lint_group_list<'a>(lints: impl Iterator<Item = &'a Lint>) -> Vec<String> {
    lints
        .map(|l| format!("        LintId::of({}::{}),", l.module, l.name.to_uppercase()))
        .sorted()
        .collect::<Vec<String>>()
}

/// Generates the `pub mod module_name` list in `clippy_lints/src/lib.rs`.
#[must_use]
pub fn gen_modules_list<'a>(lints: impl Iterator<Item = &'a Lint>) -> Vec<String> {
    lints
        .map(|l| &l.module)
        .unique()
        .map(|module| format!("mod {};", module))
        .sorted()
        .collect::<Vec<String>>()
}

/// Generates the list of lint links at the bottom of the README
#[must_use]
pub fn gen_changelog_lint_list<'a>(lints: impl Iterator<Item = &'a Lint>) -> Vec<String> {
    lints
        .sorted_by_key(|l| &l.name)
        .map(|l| format!("[`{}`]: {}#{}", l.name, DOCS_LINK, l.name))
        .collect()
}

/// Generates the `register_removed` code in `./clippy_lints/src/lib.rs`.
#[must_use]
pub fn gen_deprecated<'a>(lints: impl Iterator<Item = &'a Lint>) -> Vec<String> {
    lints
        .flat_map(|l| {
            l.deprecation
                .clone()
                .map(|depr_text| {
                    vec![
                        "    store.register_removed(".to_string(),
                        format!("        \"clippy::{}\",", l.name),
                        format!("        \"{}\",", depr_text),
                        "    );".to_string(),
                    ]
                })
                .expect("only deprecated lints should be passed")
        })
        .collect::<Vec<String>>()
}

#[must_use]
pub fn gen_register_lint_list<'a>(
    internal_lints: impl Iterator<Item = &'a Lint>,
    usable_lints: impl Iterator<Item = &'a Lint>,
) -> Vec<String> {
    let header = "    store.register_lints(&[".to_string();
    let footer = "    ]);".to_string();
    let internal_lints = internal_lints
        .sorted_by_key(|l| format!("        {}::{},", l.module, l.name.to_uppercase()))
        .map(|l| {
            format!(
                "        #[cfg(feature = \"internal-lints\")]\n        {}::{},",
                l.module,
                l.name.to_uppercase()
            )
        });
    let other_lints = usable_lints
        .sorted_by_key(|l| format!("        {}::{},", l.module, l.name.to_uppercase()))
        .map(|l| format!("        {}::{},", l.module, l.name.to_uppercase()))
        .sorted();
    let mut lint_list = vec![header];
    lint_list.extend(internal_lints);
    lint_list.extend(other_lints);
    lint_list.push(footer);
    lint_list
}

/// Gathers all files in `src/clippy_lints` and gathers all lints inside
pub fn gather_all() -> impl Iterator<Item = Lint> {
    lint_files().flat_map(|f| gather_from_file(&f))
}

fn gather_from_file(dir_entry: &walkdir::DirEntry) -> impl Iterator<Item = Lint> {
    let content = fs::read_to_string(dir_entry.path()).unwrap();
    let path = dir_entry.path();
    let filename = path.file_stem().unwrap();
    let path_buf = path.with_file_name(filename);
    let mut rel_path = path_buf
        .strip_prefix(clippy_project_root().join("clippy_lints/src"))
        .expect("only files in `clippy_lints/src` should be looked at");
    // If the lints are stored in mod.rs, we get the module name from
    // the containing directory:
    if filename == "mod" {
        rel_path = rel_path.parent().unwrap();
    }

    let module = rel_path
        .components()
        .map(|c| c.as_os_str().to_str().unwrap())
        .collect::<Vec<_>>()
        .join("::");

    parse_contents(&content, &module)
}

fn parse_contents(content: &str, module: &str) -> impl Iterator<Item = Lint> {
    let lints = DEC_CLIPPY_LINT_RE
        .captures_iter(content)
        .map(|m| Lint::new(&m["name"], &m["cat"], &m["desc"], None, module));
    let deprecated = DEC_DEPRECATED_LINT_RE
        .captures_iter(content)
        .map(|m| Lint::new(&m["name"], "Deprecated", &m["desc"], Some(&m["desc"]), module));
    // Removing the `.collect::<Vec<Lint>>().into_iter()` causes some lifetime issues due to the map
    lints.chain(deprecated).collect::<Vec<Lint>>().into_iter()
}

/// Collects all .rs files in the `clippy_lints/src` directory
fn lint_files() -> impl Iterator<Item = walkdir::DirEntry> {
    // We use `WalkDir` instead of `fs::read_dir` here in order to recurse into subdirectories.
    // Otherwise we would not collect all the lints, for example in `clippy_lints/src/methods/`.
    let path = clippy_project_root().join("clippy_lints/src");
    WalkDir::new(path)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|f| f.path().extension() == Some(OsStr::new("rs")))
}

/// Whether a file has had its text changed or not
#[derive(PartialEq, Debug)]
pub struct FileChange {
    pub changed: bool,
    pub new_lines: String,
}

/// Replaces a region in a file delimited by two lines matching regexes.
///
/// `path` is the relative path to the file on which you want to perform the replacement.
///
/// See `replace_region_in_text` for documentation of the other options.
///
/// # Panics
///
/// Panics if the path could not read or then written
pub fn replace_region_in_file<F>(
    path: &Path,
    start: &str,
    end: &str,
    replace_start: bool,
    write_back: bool,
    replacements: F,
) -> FileChange
where
    F: FnOnce() -> Vec<String>,
{
    let contents = fs::read_to_string(path).unwrap_or_else(|e| panic!("Cannot read from {}: {}", path.display(), e));
    let file_change = replace_region_in_text(&contents, start, end, replace_start, replacements);

    if write_back {
        if let Err(e) = fs::write(path, file_change.new_lines.as_bytes()) {
            panic!("Cannot write to {}: {}", path.display(), e);
        }
    }
    file_change
}

/// Replaces a region in a text delimited by two lines matching regexes.
///
/// * `text` is the input text on which you want to perform the replacement
/// * `start` is a `&str` that describes the delimiter line before the region you want to replace.
///   As the `&str` will be converted to a `Regex`, this can contain regex syntax, too.
/// * `end` is a `&str` that describes the delimiter line until where the replacement should happen.
///   As the `&str` will be converted to a `Regex`, this can contain regex syntax, too.
/// * If `replace_start` is true, the `start` delimiter line is replaced as well. The `end`
///   delimiter line is never replaced.
/// * `replacements` is a closure that has to return a `Vec<String>` which contains the new text.
///
/// If you want to perform the replacement on files instead of already parsed text,
/// use `replace_region_in_file`.
///
/// # Example
///
/// ```
/// let the_text = "replace_start\nsome text\nthat will be replaced\nreplace_end";
/// let result =
///     clippy_dev::replace_region_in_text(the_text, "replace_start", "replace_end", false, || {
///         vec!["a different".to_string(), "text".to_string()]
///     })
///     .new_lines;
/// assert_eq!("replace_start\na different\ntext\nreplace_end", result);
/// ```
///
/// # Panics
///
/// Panics if start or end is not valid regex
pub fn replace_region_in_text<F>(text: &str, start: &str, end: &str, replace_start: bool, replacements: F) -> FileChange
where
    F: FnOnce() -> Vec<String>,
{
    let replace_it = replacements();
    let mut in_old_region = false;
    let mut found = false;
    let mut new_lines = vec![];
    let start = Regex::new(start).unwrap();
    let end = Regex::new(end).unwrap();

    for line in text.lines() {
        if in_old_region {
            if end.is_match(line) {
                in_old_region = false;
                new_lines.extend(replace_it.clone());
                new_lines.push(line.to_string());
            }
        } else if start.is_match(line) {
            if !replace_start {
                new_lines.push(line.to_string());
            }
            in_old_region = true;
            found = true;
        } else {
            new_lines.push(line.to_string());
        }
    }

    if !found {
        // This happens if the provided regex in `clippy_dev/src/main.rs` does not match in the
        // given text or file. Most likely this is an error on the programmer's side and the Regex
        // is incorrect.
        eprintln!("error: regex \n{:?}\ndoesn't match. You may have to update it.", start);
        std::process::exit(1);
    }

    let mut new_lines = new_lines.join("\n");
    if text.ends_with('\n') {
        new_lines.push('\n');
    }
    let changed = new_lines != text;
    FileChange { changed, new_lines }
}

/// Returns the path to the Clippy project directory
///
/// # Panics
///
/// Panics if the current directory could not be retrieved, there was an error reading any of the
/// Cargo.toml files or ancestor directory is the clippy root directory
#[must_use]
pub fn clippy_project_root() -> PathBuf {
    let current_dir = std::env::current_dir().unwrap();
    for path in current_dir.ancestors() {
        let result = std::fs::read_to_string(path.join("Cargo.toml"));
        if let Err(err) = &result {
            if err.kind() == std::io::ErrorKind::NotFound {
                continue;
            }
        }

        let content = result.unwrap();
        if content.contains("[package]\nname = \"clippy\"") {
            return path.to_path_buf();
        }
    }
    panic!("error: Can't determine root of project. Please run inside a Clippy working dir.");
}

#[test]
fn test_parse_contents() {
    let result: Vec<Lint> = parse_contents(
        r#"
declare_clippy_lint! {
    pub PTR_ARG,
    style,
    "really long \
     text"
}

declare_clippy_lint!{
    pub DOC_MARKDOWN,
    pedantic,
    "single line"
}

/// some doc comment
declare_deprecated_lint! {
    pub SHOULD_ASSERT_EQ,
    "`assert!()` will be more flexible with RFC 2011"
}
    "#,
        "module_name",
    )
    .collect();

    let expected = vec![
        Lint::new("ptr_arg", "style", "really long text", None, "module_name"),
        Lint::new("doc_markdown", "pedantic", "single line", None, "module_name"),
        Lint::new(
            "should_assert_eq",
            "Deprecated",
            "`assert!()` will be more flexible with RFC 2011",
            Some("`assert!()` will be more flexible with RFC 2011"),
            "module_name",
        ),
    ];
    assert_eq!(expected, result);
}

#[test]
fn test_replace_region() {
    let text = "\nabc\n123\n789\ndef\nghi";
    let expected = FileChange {
        changed: true,
        new_lines: "\nabc\nhello world\ndef\nghi".to_string(),
    };
    let result = replace_region_in_text(text, r#"^\s*abc$"#, r#"^\s*def"#, false, || {
        vec!["hello world".to_string()]
    });
    assert_eq!(expected, result);
}

#[test]
fn test_replace_region_with_start() {
    let text = "\nabc\n123\n789\ndef\nghi";
    let expected = FileChange {
        changed: true,
        new_lines: "\nhello world\ndef\nghi".to_string(),
    };
    let result = replace_region_in_text(text, r#"^\s*abc$"#, r#"^\s*def"#, true, || {
        vec!["hello world".to_string()]
    });
    assert_eq!(expected, result);
}

#[test]
fn test_replace_region_no_changes() {
    let text = "123\n456\n789";
    let expected = FileChange {
        changed: false,
        new_lines: "123\n456\n789".to_string(),
    };
    let result = replace_region_in_text(text, r#"^\s*123$"#, r#"^\s*456"#, false, Vec::new);
    assert_eq!(expected, result);
}

#[test]
fn test_usable_lints() {
    let lints = vec![
        Lint::new("should_assert_eq", "Deprecated", "abc", Some("Reason"), "module_name"),
        Lint::new("should_assert_eq2", "Not Deprecated", "abc", None, "module_name"),
        Lint::new("should_assert_eq2", "internal", "abc", None, "module_name"),
        Lint::new("should_assert_eq2", "internal_style", "abc", None, "module_name"),
    ];
    let expected = vec![Lint::new(
        "should_assert_eq2",
        "Not Deprecated",
        "abc",
        None,
        "module_name",
    )];
    assert_eq!(expected, Lint::usable_lints(&lints));
}

#[test]
fn test_by_lint_group() {
    let lints = vec![
        Lint::new("should_assert_eq", "group1", "abc", None, "module_name"),
        Lint::new("should_assert_eq2", "group2", "abc", None, "module_name"),
        Lint::new("incorrect_match", "group1", "abc", None, "module_name"),
    ];
    let mut expected: HashMap<String, Vec<Lint>> = HashMap::new();
    expected.insert(
        "group1".to_string(),
        vec![
            Lint::new("should_assert_eq", "group1", "abc", None, "module_name"),
            Lint::new("incorrect_match", "group1", "abc", None, "module_name"),
        ],
    );
    expected.insert(
        "group2".to_string(),
        vec![Lint::new("should_assert_eq2", "group2", "abc", None, "module_name")],
    );
    assert_eq!(expected, Lint::by_lint_group(lints.into_iter()));
}

#[test]
fn test_gen_changelog_lint_list() {
    let lints = vec![
        Lint::new("should_assert_eq", "group1", "abc", None, "module_name"),
        Lint::new("should_assert_eq2", "group2", "abc", None, "module_name"),
    ];
    let expected = vec![
        format!("[`should_assert_eq`]: {}#should_assert_eq", DOCS_LINK.to_string()),
        format!("[`should_assert_eq2`]: {}#should_assert_eq2", DOCS_LINK.to_string()),
    ];
    assert_eq!(expected, gen_changelog_lint_list(lints.iter()));
}

#[test]
fn test_gen_deprecated() {
    let lints = vec![
        Lint::new(
            "should_assert_eq",
            "group1",
            "abc",
            Some("has been superseded by should_assert_eq2"),
            "module_name",
        ),
        Lint::new(
            "another_deprecated",
            "group2",
            "abc",
            Some("will be removed"),
            "module_name",
        ),
    ];
    let expected: Vec<String> = vec![
        "    store.register_removed(",
        "        \"clippy::should_assert_eq\",",
        "        \"has been superseded by should_assert_eq2\",",
        "    );",
        "    store.register_removed(",
        "        \"clippy::another_deprecated\",",
        "        \"will be removed\",",
        "    );",
    ]
    .into_iter()
    .map(String::from)
    .collect();
    assert_eq!(expected, gen_deprecated(lints.iter()));
}

#[test]
#[should_panic]
fn test_gen_deprecated_fail() {
    let lints = vec![Lint::new("should_assert_eq2", "group2", "abc", None, "module_name")];
    let _deprecated_lints = gen_deprecated(lints.iter());
}

#[test]
fn test_gen_modules_list() {
    let lints = vec![
        Lint::new("should_assert_eq", "group1", "abc", None, "module_name"),
        Lint::new("incorrect_stuff", "group3", "abc", None, "another_module"),
    ];
    let expected = vec!["mod another_module;".to_string(), "mod module_name;".to_string()];
    assert_eq!(expected, gen_modules_list(lints.iter()));
}

#[test]
fn test_gen_lint_group_list() {
    let lints = vec![
        Lint::new("abc", "group1", "abc", None, "module_name"),
        Lint::new("should_assert_eq", "group1", "abc", None, "module_name"),
        Lint::new("internal", "internal_style", "abc", None, "module_name"),
    ];
    let expected = vec![
        "        LintId::of(module_name::ABC),".to_string(),
        "        LintId::of(module_name::INTERNAL),".to_string(),
        "        LintId::of(module_name::SHOULD_ASSERT_EQ),".to_string(),
    ];
    assert_eq!(expected, gen_lint_group_list(lints.iter()));
}
