use std::collections::{HashMap, HashSet};

use super::{build_script_signal, build_script_stdout_signal, non_fresh_packages};
use crate::model::{BuildScriptExecutedMessage, CargoTarget, CompilerArtifactMessage};

fn build_script_msg(
    package_id: &str,
    linked_libs: &[&str],
    env: &[(&str, &str)],
) -> BuildScriptExecutedMessage {
    BuildScriptExecutedMessage {
        reason: "build-script-executed".to_string(),
        package_id: package_id.to_string(),
        linked_libs: linked_libs.iter().map(|v| (*v).to_string()).collect(),
        linked_paths: Vec::new(),
        cfgs: Vec::new(),
        env: env.iter().map(|(k, v)| ((*k).to_string(), (*v).to_string())).collect(),
        out_dir: "/tmp/out".to_string(),
    }
}

fn compiler_msg(package_id: &str, fresh: bool) -> CompilerArtifactMessage {
    CompilerArtifactMessage {
        reason: "compiler-artifact".to_string(),
        package_id: package_id.to_string(),
        manifest_path: "/tmp/Cargo.toml".to_string(),
        target: CargoTarget { kind: vec!["lib".to_string()], name: "crate".to_string() },
        profile: None,
        filenames: Vec::new(),
        executable: None,
        fresh,
    }
}

#[test]
fn build_script_signal_requires_non_fresh_support_for_confirmation() {
    let pkg = "file:///tmp/ws#0.1.0".to_string();
    let mut left = HashMap::new();
    let mut right = HashMap::new();
    left.insert(pkg.clone(), build_script_msg(&pkg, &["ssl"], &[]));
    right.insert(pkg.clone(), build_script_msg(&pkg, &["z"], &[]));

    let signal = build_script_signal(Some(&pkg), &left, &right, &HashSet::new(), &HashSet::new())
        .expect("signal");
    assert!(signal.payload_changed);
    assert!(!signal.execution_supported);
}

#[test]
fn build_script_signal_detects_env_changes() {
    let pkg = "file:///tmp/ws#0.1.0".to_string();
    let mut left = HashMap::new();
    let mut right = HashMap::new();
    left.insert(pkg.clone(), build_script_msg(&pkg, &["ssl"], &[("A", "1"), ("B", "2")]));
    right.insert(pkg.clone(), build_script_msg(&pkg, &["ssl"], &[("A", "1"), ("B", "3")]));

    let mut non_fresh = HashSet::new();
    non_fresh.insert(pkg.clone());
    let signal =
        build_script_signal(Some(&pkg), &left, &right, &non_fresh, &non_fresh).expect("signal");
    assert!(signal.env_changed);
    assert!(!signal.env_order_only);
    assert!(signal.execution_supported);
}

#[test]
fn non_fresh_package_extraction_ignores_fresh_outputs() {
    let pkgs = non_fresh_packages(&[
        compiler_msg("pkg-a", true),
        compiler_msg("pkg-b", false),
        compiler_msg("pkg-c", false),
    ]);
    assert_eq!(pkgs.len(), 2);
    assert!(pkgs.contains("pkg-b"));
    assert!(pkgs.contains("pkg-c"));
}

#[test]
fn build_script_stdout_signal_detects_order_only() {
    let pkg = "file:///tmp/ws#0.1.0".to_string();
    let mut left = HashMap::new();
    let mut right = HashMap::new();
    left.insert(
        pkg.clone(),
        vec!["cargo::rustc-link-lib=ssl".to_string(), "cargo::rustc-link-lib=z".to_string()],
    );
    right.insert(
        pkg.clone(),
        vec!["cargo::rustc-link-lib=z".to_string(), "cargo::rustc-link-lib=ssl".to_string()],
    );

    let signal = build_script_stdout_signal(Some(&pkg), &left, &right).expect("signal");
    assert!(signal.order_only);
}
