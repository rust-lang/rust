//! This is a script for validating the platform support page in the rustc book.
//!
//! The script takes two arguments, the path to the Platform Support source
//! page, and the second argument is the path to `rustc`.

use std::collections::HashSet;

fn main() {
    let mut args = std::env::args().skip(1);
    let src = args.next().expect("expected source file as first argument");
    let filename = std::path::Path::new(&src).file_name().unwrap().to_str().unwrap();
    let rustc = args.next().expect("expected rustc as second argument");
    let output = std::process::Command::new(rustc)
        .arg("--print=target-list")
        .output()
        .expect("rustc should run");
    if !output.status.success() {
        eprintln!("rustc failed to run");
        std::process::exit(0);
    }
    let stdout = std::str::from_utf8(&output.stdout).expect("utf8");
    let target_list: HashSet<_> = stdout.lines().collect();

    let doc_targets_md = std::fs::read_to_string(&src).expect("failed to read input source");
    let doc_targets: HashSet<_> = doc_targets_md
        .lines()
        .filter(|line| line.starts_with('`') && line.contains('|'))
        .map(|line| line.split('`').skip(1).next().expect("expected target code span"))
        .collect();

    let missing: Vec<_> = target_list.difference(&doc_targets).collect();
    let extra: Vec<_> = doc_targets.difference(&target_list).collect();
    for target in &missing {
        eprintln!(
            "error: target `{}` is missing from {}\n\
            If this is a new target, please add it to {}.",
            target, filename, src
        );
    }
    for target in &extra {
        eprintln!(
            "error: target `{}` is in {}, but does not appear in the rustc target list\n\
            If the target has been removed, please edit {} and remove the target.",
            target, filename, src
        );
    }
    if !missing.is_empty() || !extra.is_empty() {
        std::process::exit(1);
    }
}
