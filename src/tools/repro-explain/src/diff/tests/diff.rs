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
