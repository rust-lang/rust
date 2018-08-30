extern crate regex;
#[macro_use]
extern crate lazy_static;

use regex::Regex;
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

    pub fn active_lints(lints: &[Lint]) -> Vec<Lint> {
        lints.iter().filter(|l| l.deprecation.is_none()).cloned().collect::<Vec<Lint>>()
    }

    pub fn in_lint_group(group: &str, lints: &[Lint]) -> Vec<Lint> {
        lints.iter().filter(|l| l.group == group).cloned().collect::<Vec<Lint>>()
    }
}

pub fn collect_all() -> Vec<Lint> {
    let mut lints = vec![];
    for direntry in lint_files() {
        lints.append(&mut collect_from_file(&direntry));
    }
    lints
}

fn collect_from_file(direntry: &fs::DirEntry) -> Vec<Lint> {
    let mut file = fs::File::open(direntry.path()).unwrap();
    let mut content = String::new();
    file.read_to_string(&mut content).unwrap();
    parse_contents(&content, direntry.path().file_stem().unwrap().to_str().unwrap())
}

fn parse_contents(content: &str, filename: &str) -> Vec<Lint> {
    let mut lints: Vec<Lint> = DEC_CLIPPY_LINT_RE
        .captures_iter(&content)
        .map(|m| Lint::new(&m["name"], &m["cat"], &m["desc"], None, filename))
        .collect();
    let mut deprecated = DEC_DEPRECATED_LINT_RE
        .captures_iter(&content)
        .map(|m| Lint::new( &m["name"], "Deprecated", &m["desc"], Some(&m["desc"]), filename))
        .collect();
    lints.append(&mut deprecated);
    lints
}

/// Collects all .rs files in the `clippy_lints/src` directory
fn lint_files() -> Vec<fs::DirEntry> {
    let paths = fs::read_dir("../clippy_lints/src").unwrap();
    paths
        .filter_map(|f| f.ok())
        .filter(|f| f.path().extension() == Some(OsStr::new("rs")))
        .collect::<Vec<fs::DirEntry>>()
}

#[test]
fn test_parse_contents() {
    let result = parse_contents(
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
    "module_name");

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
    assert_eq!(expected, Lint::active_lints(&lints));
}

#[test]
fn test_in_lint_group() {
    let lints = vec![
        Lint::new("ptr_arg", "style", "really long text", None, "module_name"),
        Lint::new("doc_markdown", "pedantic", "single line", None, "module_name"),
    ];
    let expected = vec![
        Lint::new("ptr_arg", "style", "really long text", None, "module_name")
    ];
    assert_eq!(expected, Lint::in_lint_group("style", &lints));
}
