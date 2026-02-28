pub mod rules;
pub mod syn_scan;

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use crate::model::{ArtifactRecord, RuleHit};

pub fn scan_artifact_sources(artifact: &ArtifactRecord, workspace_root: &Path) -> Vec<RuleHit> {
    let mut files = BTreeSet::new();

    for input in &artifact.inputs {
        let p = PathBuf::from(input);
        if p.is_file() && p.extension().and_then(|s| s.to_str()) == Some("rs") {
            files.insert(p);
        }
    }

    if let Some(pkg) = &artifact.package_id {
        if let Some(root) = package_root_from_id(pkg) {
            let build_rs = root.join("build.rs");
            if build_rs.is_file() {
                files.insert(build_rs);
            }
        }
    }

    // If dep-info resolution was sparse, scan a few common roots as fallback.
    if files.is_empty() {
        for rel in ["build.rs", "src/lib.rs", "src/main.rs"] {
            let p = workspace_root.join(rel);
            if p.is_file() {
                files.insert(p);
            }
        }
    }

    let is_proc_macro = artifact.target_kind.iter().any(|k| k == "proc-macro");

    files
        .into_iter()
        .take(400)
        .flat_map(|path| {
            let is_build_script =
                path.file_name().and_then(|n| n.to_str()).is_some_and(|n| n == "build.rs");
            syn_scan::scan_file(&path, is_build_script, is_proc_macro)
        })
        .collect()
}

fn package_root_from_id(package_id: &str) -> Option<PathBuf> {
    let path = package_id.strip_prefix("file://")?;
    let root = path.split('#').next()?;
    let pb = PathBuf::from(root);
    if pb.is_absolute() && pb.exists() { Some(pb) } else { None }
}
