use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

pub const SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum CommandKind {
    Cargo,
    XPy,
    Other,
}

impl CommandKind {
    pub fn detect(command: &[String]) -> Self {
        let Some(bin) = command.first() else {
            return Self::Other;
        };
        if bin.ends_with("cargo") || bin == "cargo" {
            Self::Cargo
        } else if bin.ends_with("x") || bin.ends_with("x.py") || bin == "./x" || bin == "./x.py" {
            Self::XPy
        } else {
            Self::Other
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum EnvMode {
    Allowlist,
    All,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunMeta {
    pub schema_version: u32,
    pub run_id: String,
    pub command: Vec<String>,
    pub workspace_root: String,
    pub target_dir: String,
    pub env_mode: EnvMode,
    pub timestamp_utc: String,
    pub command_kind: CommandKind,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InvocationRecord {
    pub id: String,
    pub tool: String,
    pub argv: Vec<String>,
    pub cwd: String,
    pub env: BTreeMap<String, String>,
    pub crate_name: Option<String>,
    pub crate_types: Vec<String>,
    pub src_path: Option<String>,
    pub out_dir: Option<String>,
    pub dep_info: Option<String>,
    pub package_id: Option<String>,
    pub target_triple: Option<String>,
    pub profile_debuginfo: Option<String>,
    pub start_timestamp_unix: u64,
    pub end_timestamp_unix: u64,
    pub exit_code: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CargoTarget {
    pub kind: Vec<String>,
    pub name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CargoProfile {
    #[serde(default)]
    pub debuginfo: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompilerArtifactMessage {
    pub reason: String,
    pub package_id: String,
    pub manifest_path: String,
    pub target: CargoTarget,
    #[serde(default)]
    pub profile: Option<CargoProfile>,
    pub filenames: Vec<String>,
    #[serde(default)]
    pub executable: Option<String>,
    #[serde(default)]
    pub fresh: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BuildScriptExecutedMessage {
    pub reason: String,
    pub package_id: String,
    #[serde(default)]
    pub linked_libs: Vec<String>,
    #[serde(default)]
    pub linked_paths: Vec<String>,
    #[serde(default)]
    pub cfgs: Vec<String>,
    #[serde(default)]
    pub env: Vec<(String, String)>,
    pub out_dir: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BuildScriptStdoutRecord {
    pub package_id: String,
    pub lines: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactRecord {
    pub id: String,
    pub path: String,
    pub rel_path: String,
    pub kind: String,
    pub producer_invocation: Option<String>,
    #[serde(default)]
    pub producer_fingerprint: Option<String>,
    pub package_id: Option<String>,
    pub target_name: Option<String>,
    pub target_kind: Vec<String>,
    pub fresh: bool,
    pub sha256: String,
    pub inputs: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HashRecord {
    pub artifact_id: String,
    pub path: String,
    pub sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum DiffStatus {
    Identical,
    Changed,
    LeftOnly,
    RightOnly,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffEntry {
    pub artifact_key: String,
    pub status: DiffStatus,
    pub left_artifact_id: Option<String>,
    pub right_artifact_id: Option<String>,
    pub left_path: Option<String>,
    pub right_path: Option<String>,
    pub left_sha256: Option<String>,
    pub right_sha256: Option<String>,
    pub kind: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffManifest {
    pub schema_version: u32,
    pub left_run_dir: String,
    pub right_run_dir: String,
    pub entries: Vec<DiffEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceNode {
    pub id: String,
    pub kind: String,
    pub label: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceEdge {
    pub from: String,
    pub to: String,
    pub kind: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceGraph {
    pub nodes: Vec<ProvenanceNode>,
    pub edges: Vec<ProvenanceEdge>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticDiff {
    pub backend: String,
    pub summary: String,
    pub excerpt: String,
    pub left_tokens: Vec<String>,
    pub right_tokens: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "kebab-case")]
pub enum DiffClass {
    PathLeak,
    Timestamp,
    EnvLeak,
    UnstableOrder,
    BuildScript,
    ProcMacro,
    MetadataStage,
    CodegenStage,
    LinkStage,
    ScheduleSensitiveParallelism,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum StageName {
    BuildScript,
    ProcMacro,
    Metadata,
    Mir,
    LlvmIr,
    Obj,
    Link,
    Rustdoc,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceLocus {
    pub path: String,
    pub line: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Evidence {
    pub kind: String,
    pub detail: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleHit {
    pub rule_id: String,
    pub class_hint: DiffClass,
    pub path: String,
    pub line: usize,
    pub strength: String,
    pub detail: String,
    pub fix_hint: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum FindingStatus {
    Confirmed,
    StrongSuspect,
    WeakSuspect,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Finding {
    pub artifact_id: String,
    pub status: FindingStatus,
    pub class: DiffClass,
    pub first_divergent_stage: StageName,
    pub primary_locus: Option<SourceLocus>,
    pub evidence: Vec<Evidence>,
    pub fix_hint: Option<String>,
    pub score: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisFindings {
    pub schema_version: u32,
    pub findings: Vec<Finding>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayOutcome {
    pub experiment: String,
    pub success: bool,
    pub detail: String,
}
