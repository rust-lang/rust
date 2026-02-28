pub mod classify;
pub mod semantic;

use std::collections::HashSet;
use std::fs;

use anyhow::{Context, Result};
use camino::Utf8PathBuf;

use crate::capture::{load_artifacts, load_build_script_messages, load_invocation_sets};
use crate::model::{ArtifactRecord, DiffEntry, DiffManifest, DiffStatus, SCHEMA_VERSION};
use crate::provenance::build_provenance;

#[derive(Debug, Clone)]
pub struct DiffResult {
    pub analysis_dir: Utf8PathBuf,
    pub manifest: DiffManifest,
}

pub fn run_diff(
    work_dir: &Utf8PathBuf,
    left_run: &Utf8PathBuf,
    right_run: &Utf8PathBuf,
) -> Result<DiffResult> {
    let left_label =
        left_run.file_name().map(ToOwned::to_owned).unwrap_or_else(|| "left".to_string());
    let right_label =
        right_run.file_name().map(ToOwned::to_owned).unwrap_or_else(|| "right".to_string());

    let analysis_dir = work_dir.join("analysis").join(format!("{left_label}__{right_label}"));
    fs::create_dir_all(analysis_dir.join("artifacts"))
        .with_context(|| format!("failed to create analysis dir {analysis_dir}"))?;

    let left_artifacts = load_artifacts(left_run)?;
    let right_artifacts = load_artifacts(right_run)?;

    let entries = pair_and_diff(&left_artifacts, &right_artifacts);
    let manifest = DiffManifest {
        schema_version: SCHEMA_VERSION,
        left_run_dir: left_run.to_string(),
        right_run_dir: right_run.to_string(),
        entries,
    };

    write_json(analysis_dir.join("diff-manifest.json"), &manifest)?;

    let left_invocations = load_invocation_sets(left_run)?;
    let right_invocations = load_invocation_sets(right_run)?;
    let left_build_scripts = load_build_script_messages(left_run).unwrap_or_default();
    let right_build_scripts = load_build_script_messages(right_run).unwrap_or_default();
    let provenance = build_provenance(
        &left_label,
        &right_label,
        &left_artifacts,
        &right_artifacts,
        &left_build_scripts,
        &right_build_scripts,
        &left_invocations,
        &right_invocations,
    );
    write_json(analysis_dir.join("provenance.json"), &provenance)?;

    Ok(DiffResult { analysis_dir, manifest })
}

fn pair_and_diff(left: &[ArtifactRecord], right: &[ArtifactRecord]) -> Vec<DiffEntry> {
    let mut entries = Vec::new();
    let mut used_right = HashSet::new();

    for l in left {
        let key = artifact_pair_key(l);
        let match_idx = find_match_index(l, right, &used_right);

        if let Some(idx) = match_idx {
            used_right.insert(idx);
            let r = &right[idx];
            let status =
                if l.sha256 == r.sha256 { DiffStatus::Identical } else { DiffStatus::Changed };
            entries.push(DiffEntry {
                artifact_key: key,
                status,
                left_artifact_id: Some(l.id.clone()),
                right_artifact_id: Some(r.id.clone()),
                left_path: Some(l.path.clone()),
                right_path: Some(r.path.clone()),
                left_sha256: Some(l.sha256.clone()),
                right_sha256: Some(r.sha256.clone()),
                kind: l.kind.clone(),
            });
        } else {
            entries.push(DiffEntry {
                artifact_key: key,
                status: DiffStatus::LeftOnly,
                left_artifact_id: Some(l.id.clone()),
                right_artifact_id: None,
                left_path: Some(l.path.clone()),
                right_path: None,
                left_sha256: Some(l.sha256.clone()),
                right_sha256: None,
                kind: l.kind.clone(),
            });
        }
    }

    for (idx, r) in right.iter().enumerate() {
        if used_right.contains(&idx) {
            continue;
        }
        entries.push(DiffEntry {
            artifact_key: artifact_pair_key(r),
            status: DiffStatus::RightOnly,
            left_artifact_id: None,
            right_artifact_id: Some(r.id.clone()),
            left_path: None,
            right_path: Some(r.path.clone()),
            left_sha256: None,
            right_sha256: Some(r.sha256.clone()),
            kind: r.kind.clone(),
        });
    }

    entries.sort_by(|a, b| a.artifact_key.cmp(&b.artifact_key));
    entries
}

fn find_match_index(
    left: &ArtifactRecord,
    right: &[ArtifactRecord],
    used_right: &HashSet<usize>,
) -> Option<usize> {
    let left_primary = compiler_artifact_pair_key(left);
    if left_primary.is_some() {
        if let Some(idx) = right
            .iter()
            .enumerate()
            .find(|(idx, r)| {
                if used_right.contains(idx) {
                    return false;
                }
                compiler_artifact_pair_key(r) == left_primary
            })
            .map(|(idx, _)| idx)
        {
            return Some(idx);
        }
    }

    if let Some(idx) = right
        .iter()
        .enumerate()
        .find(|(idx, r)| {
            if used_right.contains(idx) {
                return false;
            }
            !left.rel_path.is_empty() && left.rel_path == r.rel_path
        })
        .map(|(idx, _)| idx)
    {
        return Some(idx);
    }

    let Some(left_fp) = left.producer_fingerprint.as_ref() else {
        return None;
    };
    right
        .iter()
        .enumerate()
        .find(|(idx, r)| {
            if used_right.contains(idx) {
                return false;
            }
            r.kind == left.kind && r.producer_fingerprint.as_ref() == Some(left_fp)
        })
        .map(|(idx, _)| idx)
}

fn artifact_pair_key(artifact: &ArtifactRecord) -> String {
    if let Some(key) = compiler_artifact_pair_key(artifact) {
        key
    } else if !artifact.rel_path.is_empty() {
        format!("rel={}|artifact_kind={}", artifact.rel_path, artifact.kind)
    } else if let Some(fp) = &artifact.producer_fingerprint {
        format!("inv-fp={fp}|artifact_kind={}", artifact.kind)
    } else {
        format!("path={}|artifact_kind={}", artifact.path, artifact.kind)
    }
}

fn compiler_artifact_pair_key(artifact: &ArtifactRecord) -> Option<String> {
    let basename = std::path::Path::new(&artifact.path)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("<unknown>");

    let (Some(pkg), Some(target_name)) = (&artifact.package_id, &artifact.target_name) else {
        return None;
    };
    let target_kind = if artifact.target_kind.is_empty() {
        "<unknown>".to_string()
    } else {
        artifact.target_kind.join(",")
    };
    Some(format!(
        "pkg={pkg}|target={target_name}|kind={target_kind}|basename={basename}|artifact_kind={}",
        artifact.kind
    ))
}

pub fn load_manifest(analysis_dir: &Utf8PathBuf) -> Result<DiffManifest> {
    let data = fs::read(analysis_dir.join("diff-manifest.json"))
        .with_context(|| format!("failed to read {analysis_dir}/diff-manifest.json"))?;
    serde_json::from_slice(&data).context("failed to parse diff-manifest.json")
}

fn write_json(path: Utf8PathBuf, value: &impl serde::Serialize) -> Result<()> {
    let body = serde_json::to_vec_pretty(value).context("failed to serialize json")?;
    fs::write(&path, body).with_context(|| format!("failed to write {path}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::pair_and_diff;
    use crate::model::{ArtifactRecord, DiffStatus};

    fn artifact(
        id: &str,
        path: &str,
        rel_path: &str,
        package_id: Option<&str>,
        target_name: Option<&str>,
        target_kind: &[&str],
        producer_fingerprint: Option<&str>,
        sha256: &str,
    ) -> ArtifactRecord {
        ArtifactRecord {
            id: id.to_string(),
            path: path.to_string(),
            rel_path: rel_path.to_string(),
            kind: "rmeta".to_string(),
            producer_invocation: None,
            producer_fingerprint: producer_fingerprint.map(str::to_string),
            package_id: package_id.map(str::to_string),
            target_name: target_name.map(str::to_string),
            target_kind: target_kind.iter().map(|s| (*s).to_string()).collect(),
            fresh: false,
            sha256: sha256.to_string(),
            inputs: Vec::new(),
        }
    }

    #[test]
    fn pair_uses_rel_path_when_primary_key_missing() {
        let left = vec![artifact(
            "art-left",
            "/left/target/foo.rmeta",
            "target/foo.rmeta",
            None,
            None,
            &[],
            None,
            "aaa",
        )];
        let right = vec![artifact(
            "art-right",
            "/right/target/foo.rmeta",
            "target/foo.rmeta",
            None,
            None,
            &[],
            None,
            "bbb",
        )];

        let entries = pair_and_diff(&left, &right);
        assert_eq!(entries.len(), 1);
        assert!(matches!(entries[0].status, DiffStatus::Changed));
        assert_eq!(entries[0].left_artifact_id.as_deref(), Some("art-left"));
        assert_eq!(entries[0].right_artifact_id.as_deref(), Some("art-right"));
    }

    #[test]
    fn pair_falls_back_to_producer_fingerprint() {
        let left = vec![artifact(
            "art-left",
            "/left/other/a.rmeta",
            "a.rmeta",
            None,
            None,
            &[],
            Some("fp:crate-a"),
            "same",
        )];
        let right = vec![artifact(
            "art-right",
            "/right/different/b.rmeta",
            "b.rmeta",
            None,
            None,
            &[],
            Some("fp:crate-a"),
            "same",
        )];

        let entries = pair_and_diff(&left, &right);
        assert_eq!(entries.len(), 1);
        assert!(matches!(entries[0].status, DiffStatus::Identical));
        assert_eq!(entries[0].left_artifact_id.as_deref(), Some("art-left"));
        assert_eq!(entries[0].right_artifact_id.as_deref(), Some("art-right"));
    }
}
