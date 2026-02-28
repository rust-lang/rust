use super::{build_script_instruction_diff, with_focused_package, with_jobs, with_jobs_one};
use crate::model::BuildScriptStdoutRecord;

#[test]
fn injects_jobs_when_missing() {
    let cmd = vec!["cargo".to_string(), "build".to_string()];
    let got = with_jobs_one(&cmd);
    assert_eq!(got, vec!["cargo", "build", "--jobs", "1"]);
}

#[test]
fn rewrites_jobs_equals_style() {
    let cmd = vec!["cargo".to_string(), "build".to_string(), "--jobs=8".to_string()];
    let got = with_jobs_one(&cmd);
    assert_eq!(got, vec!["cargo", "build", "--jobs=1"]);
}

#[test]
fn rewrites_short_style() {
    let cmd = vec!["cargo".to_string(), "build".to_string(), "-j16".to_string()];
    let got = with_jobs_one(&cmd);
    assert_eq!(got, vec!["cargo", "build", "-j1"]);
}

#[test]
fn rewrites_jobs_to_requested_value() {
    let cmd = vec!["cargo".to_string(), "build".to_string(), "--jobs=8".to_string()];
    let got = with_jobs(&cmd, 4);
    assert_eq!(got, vec!["cargo", "build", "--jobs=4"]);
}

#[test]
fn injects_package_filter_for_cargo_when_missing() {
    let cmd = vec!["cargo".to_string(), "build".to_string(), "--release".to_string()];
    let got = with_focused_package(&cmd, "file:///tmp/ws#0.1.0");
    assert_eq!(got, vec!["cargo", "build", "--release", "--package", "file:///tmp/ws#0.1.0"]);
}

#[test]
fn keeps_existing_package_filter() {
    let cmd = vec![
        "cargo".to_string(),
        "build".to_string(),
        "--package".to_string(),
        "crate-a".to_string(),
    ];
    let got = with_focused_package(&cmd, "crate-b");
    assert_eq!(got, cmd);
}

#[test]
fn build_script_instruction_diff_detects_order_only_changes() {
    let left = vec![BuildScriptStdoutRecord {
        package_id: "pkg".to_string(),
        lines: vec!["cargo::rustc-link-lib=ssl".to_string(), "cargo::rustc-link-lib=z".to_string()],
    }];
    let right = vec![BuildScriptStdoutRecord {
        package_id: "pkg".to_string(),
        lines: vec!["cargo::rustc-link-lib=z".to_string(), "cargo::rustc-link-lib=ssl".to_string()],
    }];

    let (changed, order_only) = build_script_instruction_diff(&left, &right, Some("pkg"));
    assert!(changed);
    assert!(order_only);
}

#[test]
fn build_script_instruction_diff_treats_missing_side_as_non_order_only() {
    let left = vec![BuildScriptStdoutRecord {
        package_id: "pkg".to_string(),
        lines: vec!["cargo::rustc-link-lib=ssl".to_string()],
    }];
    let (changed, order_only) = build_script_instruction_diff(&left, &[], Some("pkg"));
    assert!(changed);
    assert!(!order_only);
}
