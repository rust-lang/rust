use std::collections::HashMap;

use regex::Regex;

use crate::model::{
    ArtifactRecord, DiffClass, Evidence, FindingStatus, RuleHit, SemanticDiff, StageName,
};

#[derive(Debug, Clone)]
pub struct Classification {
    pub class: DiffClass,
    pub stage: StageName,
    pub evidence: Vec<Evidence>,
}

pub fn classify_artifact(
    artifact: &ArtifactRecord,
    semantic: &SemanticDiff,
    rule_hits: &[RuleHit],
) -> Classification {
    let mut evidence = Vec::new();

    let has_path_leak = semantic.excerpt.contains("<run-root>")
        || semantic.excerpt.contains("/home/")
        || semantic.excerpt.contains("\\\\");
    if has_path_leak {
        evidence.push(Evidence {
            kind: "semantic-diff".to_string(),
            detail: "path-like strings appear in diff".to_string(),
        });
    }

    let ts_re = Regex::new(r"(20\d\d-\d\d-\d\d[T ]\d\d:\d\d:\d\d|\b1\d{9}\b)").ok();
    let has_timestamp = ts_re.as_ref().is_some_and(|re| re.is_match(&semantic.excerpt));
    if has_timestamp {
        evidence.push(Evidence {
            kind: "semantic-diff".to_string(),
            detail: "timestamp-like token detected".to_string(),
        });
    }

    let env_hint = semantic.excerpt.contains("PATH")
        || semantic.excerpt.contains("HOME")
        || semantic.excerpt.contains("USERPROFILE")
        || rule_hits.iter().any(|r| r.class_hint == DiffClass::EnvLeak);
    if env_hint {
        evidence.push(Evidence {
            kind: "semantic-diff".to_string(),
            detail: "environment-related content detected".to_string(),
        });
    }

    let order_only = is_order_only(semantic);
    if order_only {
        evidence.push(Evidence {
            kind: "semantic-diff".to_string(),
            detail: "token multiset matches; order differs".to_string(),
        });
    }

    let build_script_related = artifact.kind == "out-dir-file"
        || rule_hits.iter().any(|r| r.rule_id == "RE005" || r.rule_id == "RE006");
    if build_script_related {
        evidence.push(Evidence {
            kind: "rule-hit".to_string(),
            detail: "build script related rule triggered".to_string(),
        });
    }

    let proc_macro_related = artifact.target_kind.iter().any(|k| k == "proc-macro")
        || rule_hits.iter().any(|r| r.rule_id == "RE007");

    let parallel_related = rule_hits.iter().any(|r| r.rule_id == "RE008");

    let class = if has_timestamp {
        DiffClass::Timestamp
    } else if build_script_related {
        DiffClass::BuildScript
    } else if proc_macro_related {
        DiffClass::ProcMacro
    } else if parallel_related {
        DiffClass::ScheduleSensitiveParallelism
    } else if order_only {
        DiffClass::UnstableOrder
    } else if env_hint {
        DiffClass::EnvLeak
    } else if has_path_leak {
        DiffClass::PathLeak
    } else {
        stage_class_from_kind(&artifact.kind)
    };

    let stage = first_stage_from_kind(&artifact.kind, &class);

    Classification { class, stage, evidence }
}

fn stage_class_from_kind(kind: &str) -> DiffClass {
    match kind {
        "rmeta" => DiffClass::MetadataStage,
        "obj" | "llvm-ir" | "llvm-bc" => DiffClass::CodegenStage,
        "rustdoc-html" => DiffClass::LinkStage,
        "binary" | "dylib" | "rlib" | "staticlib" => DiffClass::LinkStage,
        _ => DiffClass::Unknown,
    }
}

pub fn first_stage_from_kind(kind: &str, class: &DiffClass) -> StageName {
    if class == &DiffClass::BuildScript {
        return StageName::BuildScript;
    }
    if class == &DiffClass::ProcMacro {
        return StageName::ProcMacro;
    }

    match kind {
        "rmeta" => StageName::Metadata,
        "llvm-ir" | "llvm-bc" => StageName::LlvmIr,
        "obj" => StageName::Obj,
        "rustdoc-html" => StageName::Rustdoc,
        "out-dir-file" => StageName::BuildScript,
        "binary" | "dylib" | "rlib" | "staticlib" => StageName::Link,
        _ => StageName::Unknown,
    }
}

pub fn compute_score(
    class: &DiffClass,
    stage: &StageName,
    semantic: &SemanticDiff,
    rule_hits: &[RuleHit],
    replay_confirmed: bool,
    direct_build_script: bool,
) -> (i32, FindingStatus) {
    let class_match = if *class == DiffClass::Unknown { 0 } else { 3 };
    let locality_match = if rule_hits.is_empty() { 0 } else { 2 };
    let stage_match = if *stage == StageName::Unknown { 0 } else { 3 };
    let textual_match = if semantic.excerpt.is_empty() {
        0
    } else if semantic.excerpt.len() > 32 {
        2
    } else {
        1
    };
    let replay_bonus = if replay_confirmed { 5 } else { 0 };
    let direct_build_script_bonus = if direct_build_script { 4 } else { 0 };

    let score = class_match
        + locality_match
        + stage_match
        + textual_match
        + replay_bonus
        + direct_build_script_bonus;

    let status = if replay_bonus > 0 || (direct_build_script_bonus > 0 && score >= 8) {
        FindingStatus::Confirmed
    } else if score >= 5 {
        FindingStatus::StrongSuspect
    } else {
        FindingStatus::WeakSuspect
    };

    (score, status)
}

fn is_order_only(semantic: &SemanticDiff) -> bool {
    if semantic.left_tokens.is_empty() || semantic.right_tokens.is_empty() {
        return false;
    }
    if semantic.left_tokens == semantic.right_tokens {
        return false;
    }
    same_multiset(&semantic.left_tokens, &semantic.right_tokens)
}

fn same_multiset(left: &[String], right: &[String]) -> bool {
    if left.len() != right.len() {
        return false;
    }
    let mut counts = HashMap::<&str, i32>::new();
    for t in left {
        *counts.entry(t).or_insert(0) += 1;
    }
    for t in right {
        let Some(entry) = counts.get_mut(t.as_str()) else {
            return false;
        };
        *entry -= 1;
        if *entry < 0 {
            return false;
        }
    }
    counts.values().all(|v| *v == 0)
}
