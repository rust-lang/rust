use std::fs;

use camino::Utf8PathBuf;
use tempfile::tempdir;

use super::{collect_build_script_stdout_records, copy_cargo_timings, normalize_cargo_instruction};
use crate::model::BuildScriptExecutedMessage;

#[test]
fn normalizes_both_cargo_instruction_prefixes() {
    assert_eq!(
        normalize_cargo_instruction("cargo:rustc-link-lib=z"),
        Some("cargo::rustc-link-lib=z".to_string())
    );
    assert_eq!(
        normalize_cargo_instruction("cargo::rustc-link-lib=z"),
        Some("cargo::rustc-link-lib=z".to_string())
    );
    assert_eq!(normalize_cargo_instruction("warning: hello"), None);
}

#[test]
fn collects_build_script_stdout_records_from_output_file() {
    let dir = tempdir().expect("tempdir");
    let build_dir = dir.path().join("build-script-abc");
    let out_dir = build_dir.join("out");
    fs::create_dir_all(&out_dir).expect("create out_dir");
    fs::write(
        build_dir.join("output"),
        "cargo:rustc-link-lib=ssl\ncargo::rustc-link-lib=z\nnot-cargo-line\n",
    )
    .expect("write output");

    let msgs = vec![BuildScriptExecutedMessage {
        reason: "build-script-executed".to_string(),
        package_id: "pkg".to_string(),
        linked_libs: Vec::new(),
        linked_paths: Vec::new(),
        cfgs: Vec::new(),
        env: Vec::new(),
        out_dir: out_dir.to_string_lossy().into_owned(),
    }];

    let records = collect_build_script_stdout_records(&msgs);
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].package_id, "pkg");
    assert_eq!(
        records[0].lines,
        vec!["cargo::rustc-link-lib=ssl".to_string(), "cargo::rustc-link-lib=z".to_string(),]
    );
}

#[test]
fn copies_cargo_timings_tree_when_present() {
    let dir = tempdir().expect("tempdir");
    let target_dir = dir.path().join("target");
    let src = target_dir.join("cargo-timings").join("timing-report.html");
    fs::create_dir_all(src.parent().expect("parent")).expect("create cargo timings dir");
    fs::write(&src, "<html>timings</html>").expect("write timing report");

    let run_timings_dir =
        Utf8PathBuf::from_path_buf(dir.path().join("run").join("timings")).expect("utf8 path");
    fs::create_dir_all(&run_timings_dir).expect("create run timings dir");

    copy_cargo_timings(&target_dir, &run_timings_dir).expect("copy timings");
    let copied = run_timings_dir.join("timing-report.html");
    assert!(copied.is_file(), "copied timing report should exist");
    assert_eq!(fs::read_to_string(copied).expect("read copied"), "<html>timings</html>");
}
