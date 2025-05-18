// This test checks that all symbols defined in Clippy's `sym.rs` file
// are used in Clippy. Otherwise, it will fail with a list of symbols
// which are unused.
//
// This test is a no-op if run as part of the compiler test suite
// and will always succeed.

use std::collections::HashSet;
use std::ffi::OsStr;
use std::fs;

use regex::Regex;
use walkdir::{DirEntry, WalkDir};

const SYM_FILE: &str = "clippy_utils/src/sym.rs";

type Result<T, E = AnyError> = std::result::Result<T, E>;
type AnyError = Box<dyn std::error::Error>;

#[test]
#[allow(clippy::case_sensitive_file_extension_comparisons)]
fn all_symbols_are_used() -> Result<()> {
    if option_env!("RUSTC_TEST_SUITE").is_some() {
        return Ok(());
    }

    // Load all symbols defined in `SYM_FILE`.
    let content = fs::read_to_string(SYM_FILE)?;
    let content = content
        .split_once("generate! {")
        .ok_or("cannot find symbols start")?
        .1
        .split_once("\n}\n")
        .ok_or("cannot find symbols end")?
        .0;
    let mut interned: HashSet<String> = Regex::new(r"(?m)^    (\w+)")
        .unwrap()
        .captures_iter(content)
        .map(|m| m[1].to_owned())
        .collect();

    // Remove symbols used as `sym::*`.
    let used_re = Regex::new(r"\bsym::(\w+)\b").unwrap();
    let rs_ext = OsStr::new("rs");
    for dir in ["clippy_lints", "clippy_lints_internal", "clippy_utils", "src"] {
        for file in WalkDir::new(dir)
            .into_iter()
            .flatten()
            .map(DirEntry::into_path)
            .filter(|p| p.extension() == Some(rs_ext))
        {
            for cap in used_re.captures_iter(&fs::read_to_string(file)?) {
                interned.remove(&cap[1]);
            }
        }
    }

    // Remove symbols used as part of paths.
    let paths_re = Regex::new(r"path!\(([\w:]+)\)").unwrap();
    for path in [
        "clippy_utils/src/paths.rs",
        "clippy_lints_internal/src/internal_paths.rs",
    ] {
        for cap in paths_re.captures_iter(&fs::read_to_string(path)?) {
            for sym in cap[1].split("::") {
                interned.remove(sym);
            }
        }
    }

    let mut extra = interned.iter().collect::<Vec<_>>();
    if !extra.is_empty() {
        extra.sort_unstable();
        eprintln!("Unused symbols defined in {SYM_FILE}:");
        for sym in extra {
            eprintln!("  - {sym}");
        }
        Err(format!("extra symbols found — remove them {SYM_FILE}"))?;
    }
    Ok(())
}
