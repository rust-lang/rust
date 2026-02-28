use camino::Utf8PathBuf;
use tempfile::tempdir;

use super::{localize_first_divergent_stage, map_proc_macro_stage};
use crate::model::{BuildScriptExecutedMessage, InvocationRecord, StageName};

fn inv(id: &str, tool: &str) -> InvocationRecord {
    InvocationRecord {
        id: id.to_string(),
        tool: tool.to_string(),
        argv: Vec::new(),
        cwd: ".".to_string(),
        env: std::collections::BTreeMap::new(),
        crate_name: None,
        crate_types: Vec::new(),
        src_path: None,
        out_dir: None,
        dep_info: None,
        package_id: None,
        target_triple: None,
        profile_debuginfo: None,
        start_timestamp_unix: 0,
        end_timestamp_unix: 0,
        exit_code: 0,
    }
}

fn build_script(pkg: &str, out_dir: &str, libs: &[&str]) -> BuildScriptExecutedMessage {
    BuildScriptExecutedMessage {
        reason: "build-script-executed".to_string(),
        package_id: pkg.to_string(),
        linked_libs: libs.iter().map(|v| (*v).to_string()).collect(),
        linked_paths: Vec::new(),
        cfgs: Vec::new(),
        env: Vec::new(),
        out_dir: out_dir.to_string(),
    }
}

#[test]
fn localize_reports_build_script_stage_before_rustc_replay() {
    let left = inv("left", "not-rustc");
    let right = inv("right", "not-rustc");
    let dir = tempdir().expect("tempdir");
    let scratch = Utf8PathBuf::from_path_buf(dir.path().to_path_buf()).expect("utf8");

    let left_build = build_script("pkg", "/tmp/a", &["ssl"]);
    let right_build = build_script("pkg", "/tmp/b", &["z"]);

    let loc = localize_first_divergent_stage(
        &left,
        &right,
        Some(&left_build),
        Some(&right_build),
        &scratch,
    )
    .expect("localization");

    assert_eq!(loc.first_divergent_stage, StageName::BuildScript);
    assert!(loc.checks.iter().any(|check| check.stage == StageName::BuildScript));
}

#[test]
fn proc_macro_mapping_applies_only_to_early_codegen_stages() {
    assert_eq!(map_proc_macro_stage(StageName::Metadata, true), StageName::ProcMacro);
    assert_eq!(map_proc_macro_stage(StageName::Obj, true), StageName::ProcMacro);
    assert_eq!(map_proc_macro_stage(StageName::Link, true), StageName::Link);
    assert_eq!(map_proc_macro_stage(StageName::Metadata, false), StageName::Metadata);
}
