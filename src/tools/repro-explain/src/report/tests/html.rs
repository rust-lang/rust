use super::{collect_run_roots, redact_for_report};
use crate::model::DiffManifest;

#[test]
fn redacts_run_and_workspace_roots() {
    let run_roots = vec!["/tmp/work/.repro/runs/A".to_string()];
    let workspace_roots = vec!["/tmp/work".to_string()];
    let value = "/tmp/work/.repro/runs/A/target/debug/deps";
    let redacted = redact_for_report(value, &run_roots, &workspace_roots);
    assert!(redacted.contains("<run-root>"));
    assert!(!redacted.contains("/tmp/work/.repro/runs/A"));
}

#[test]
fn collect_run_roots_dedups_and_keeps_longer_first() {
    let manifest = DiffManifest {
        schema_version: 1,
        left_run_dir: "/tmp/work/.repro/runs/A".to_string(),
        right_run_dir: "/tmp/work/.repro/runs/B".to_string(),
        entries: Vec::new(),
    };
    let roots = collect_run_roots(&manifest);
    assert_eq!(roots.len(), 2);
    assert_eq!(roots[0], "/tmp/work/.repro/runs/A");
}
