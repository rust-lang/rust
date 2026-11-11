use std::fs;
use std::path::Path;
use std::process::Command;
use std::sync::LazyLock;

use pretty_assertions::assert_str_eq;
use regex::Regex;
use update_api_list::WORKSPACE_ROOT;

static PUBLIC_FUNCTIONS: LazyLock<Vec<String>> = LazyLock::new(|| {
    fs::read_to_string(WORKSPACE_ROOT.join("etc/function-list.txt"))
        .unwrap()
        .lines()
        .map(|line| line.trim())
        .filter(|line| !(line.starts_with("#") || line.is_empty()))
        .map(|line| line.to_owned())
        .collect()
});

/// In each file, check annotations indicating that blocks of code should be sorted or should
/// include an exhaustive list of all public API.
#[test]
fn tidy_lists() {
    let out = Command::new("git")
        .arg("ls-files")
        .current_dir(&*WORKSPACE_ROOT)
        .output()
        .unwrap();
    assert!(out.status.success());

    let file_list = str::from_utf8(&out.stdout).unwrap();

    for path in file_list.lines() {
        let relpath = Path::new(path);
        let abspath = WORKSPACE_ROOT.join(relpath);
        if abspath.is_dir() || relpath == file!() {
            continue;
        }

        let src = fs::read_to_string(&abspath).unwrap();
        let lines: Vec<_> = src.lines().collect();

        validate_delimited_block(
            relpath,
            &lines,
            "verify-sorted-start",
            "verify-sorted-end",
            ensure_sorted,
        );

        validate_delimited_block(
            relpath,
            &lines,
            "verify-apilist-start",
            "verify-apilist-end",
            ensure_contains_api,
        );
    }
}

/// Identify blocks of code wrapped within `start` and `end`, collect their contents to a list of
/// strings, and call `validate` for each of those lists.
fn validate_delimited_block(
    relpath: &Path,
    lines: &[&str],
    start: &str,
    end: &str,
    validate: impl Fn(&Path, usize, &[&str]),
) {
    let mut block_lines = Vec::new();
    let mut block_start_line = None;
    for (mut line_num, line) in lines.iter().enumerate() {
        line_num += 1;

        if line.contains(start) {
            block_start_line = Some(line_num);
            continue;
        }

        // End of a block, validate its contents
        if line.contains(end) {
            let Some(start_line) = block_start_line else {
                panic!("`{end}` without `{start}` at {relpath:?}:{line_num}");
            };

            validate(relpath, start_line, &block_lines);
            block_lines.clear();
            block_start_line = None;
            continue;
        }

        if block_start_line.is_some() {
            block_lines.push(*line);
        }
    }

    if let Some(start_line) = block_start_line {
        panic!("`{start}` without `{end}` at {relpath:?}:{start_line}");
    }
}

/// Given a list of strings, ensure that each public function we have is named somewhere.
fn ensure_contains_api(relpath: &Path, block_start_line: usize, lines: &[&str]) {
    let mut not_found = Vec::new();

    for func in &*PUBLIC_FUNCTIONS {
        // The function name may be on its own or somewhere in a snake case string.
        let re = Regex::new(&format!(r"(\b|_){func}(\b|_)")).unwrap();
        if !lines.iter().any(|line| re.is_match(line)) {
            not_found.push(func);
        }
    }

    if not_found.is_empty() {
        return;
    }

    panic!("functions not found at {relpath:?}:{block_start_line}: {not_found:?}");
}

fn ensure_sorted(relpath: &Path, block_start_line: usize, lines: &[&str]) {
    let mut sorted = lines.to_owned();
    sorted.sort_unstable();
    let a = lines.join("\n");
    let b = sorted.join("\n");

    assert_str_eq!(a, b, "sorted block at {relpath:?}:{block_start_line}");
}
