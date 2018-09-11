#![feature(tool_lints)]
#![allow(clippy::default_hash_types)]
extern crate regex;
#[macro_use]
extern crate lazy_static;
extern crate itertools;

use regex::Regex;
use itertools::Itertools;
use std::collections::HashMap;
use std::ffi::OsStr;
use std::fs;
use std::io::prelude::*;

lazy_static! {
    static ref DEC_CLIPPY_LINT_RE: Regex = Regex::new(r#"(?x)
        declare_clippy_lint!\s*[\{(]\s*
        pub\s+(?P<name>[A-Z_][A-Z_0-9]*)\s*,\s*
        (?P<cat>[a-z_]+)\s*,\s*
        "(?P<desc>(?:[^"\\]+|\\(?s).(?-s))*)"\s*[})]
    "#).unwrap();
    static ref DEC_DEPRECATED_LINT_RE: Regex = Regex::new(r#"(?x)
        declare_deprecated_lint!\s*[{(]\s*
        pub\s+(?P<name>[A-Z_][A-Z_0-9]*)\s*,\s*
        "(?P<desc>(?:[^"\\]+|\\(?s).(?-s))*)"\s*[})]
    "#).unwrap();
    static ref NL_ESCAPE_RE: Regex = Regex::new(r#"\\\n\s*"#).unwrap();
    pub static ref DOCS_LINK: String = "https://rust-lang-nursery.github.io/rust-clippy/master/index.html".to_string();
}

#[derive(Clone, PartialEq, Debug)]
pub struct Lint {
    pub name: String,
    pub group: String,
    pub desc: String,
    pub deprecation: Option<String>,
    pub module: String,
}

impl Lint {
    pub fn new(name: &str, group: &str, desc: &str, deprecation: Option<&str>, module: &str) -> Lint {
        Lint {
            name: name.to_lowercase(),
            group: group.to_string(),
            desc: NL_ESCAPE_RE.replace(&desc.replace("\\\"", "\""), "").to_string(),
            deprecation: deprecation.map(|d| d.to_string()),
            module: module.to_string(),
        }
    }

    /// Returns all non-deprecated lints
    pub fn active_lints(lints: &[Lint]) -> impl Iterator<Item=&Lint> {
        lints.iter().filter(|l| l.deprecation.is_none())
    }

    /// Returns the lints in a HashMap, grouped by the different lint groups
    pub fn by_lint_group(lints: &[Lint]) -> HashMap<String, Vec<Lint>> {
        lints.iter().map(|lint| (lint.group.to_string(), lint.clone())).into_group_map()
    }
}

pub fn gather_all() -> impl Iterator<Item=Lint> {
    lint_files().flat_map(|f| gather_from_file(&f))
}

fn gather_from_file(dir_entry: &fs::DirEntry) -> impl Iterator<Item=Lint> {
    let mut file = fs::File::open(dir_entry.path()).unwrap();
    let mut content = String::new();
    file.read_to_string(&mut content).unwrap();
    parse_contents(&content, dir_entry.path().file_stem().unwrap().to_str().unwrap())
}

fn parse_contents(content: &str, filename: &str) -> impl Iterator<Item=Lint> {
    let lints = DEC_CLIPPY_LINT_RE
        .captures_iter(content)
        .map(|m| Lint::new(&m["name"], &m["cat"], &m["desc"], None, filename));
    let deprecated = DEC_DEPRECATED_LINT_RE
        .captures_iter(content)
        .map(|m| Lint::new( &m["name"], "Deprecated", &m["desc"], Some(&m["desc"]), filename));
    // Removing the `.collect::<Vec<Lint>>().into_iter()` causes some lifetime issues due to the map
    lints.chain(deprecated).collect::<Vec<Lint>>().into_iter()
}

/// Collects all .rs files in the `clippy_lints/src` directory
fn lint_files() -> impl Iterator<Item=fs::DirEntry> {
    fs::read_dir("../clippy_lints/src")
        .unwrap()
        .filter_map(|f| f.ok())
        .filter(|f| f.path().extension() == Some(OsStr::new("rs")))
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
    "module_name").collect();

    let expected = vec![
        Lint::new("ptr_arg", "style", "really long text", None, "module_name"),
        Lint::new("doc_markdown", "pedantic", "single line", None, "module_name"),
        Lint::new(
            "should_assert_eq",
            "Deprecated",
            "`assert!()` will be more flexible with RFC 2011",
            Some("`assert!()` will be more flexible with RFC 2011"),
            "module_name"
        ),
    ];
    assert_eq!(expected, result);
}

#[test]
fn test_active_lints() {
    let lints = vec![
        Lint::new("should_assert_eq", "Deprecated", "abc", Some("Reason"), "module_name"),
        Lint::new("should_assert_eq2", "Not Deprecated", "abc", None, "module_name")
    ];
    let expected = vec![
        Lint::new("should_assert_eq2", "Not Deprecated", "abc", None, "module_name")
    ];
    assert_eq!(expected, Lint::active_lints(&lints).cloned().collect::<Vec<Lint>>());
}

#[test]
fn test_by_lint_group() {
    let lints = vec![
        Lint::new("should_assert_eq", "group1", "abc", None, "module_name"),
        Lint::new("should_assert_eq2", "group2", "abc", None, "module_name"),
        Lint::new("incorrect_match", "group1", "abc", None, "module_name"),
    ];
    let mut expected: HashMap<String, Vec<Lint>> = HashMap::new();
    expected.insert("group1".to_string(), vec![
        Lint::new("should_assert_eq", "group1", "abc", None, "module_name"),
        Lint::new("incorrect_match", "group1", "abc", None, "module_name"),
    ]);
    expected.insert("group2".to_string(), vec![
        Lint::new("should_assert_eq2", "group2", "abc", None, "module_name")
    ]);
    assert_eq!(expected, Lint::by_lint_group(&lints));
}
