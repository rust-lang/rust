use serde_derive::{Deserialize, Serialize};

/// The default set of crates for opt-dist to collect LLVM profiles.
pub const LLVM_PGO_CRATES: &[&str] = &[
    "syn-1.0.89",
    "cargo-0.60.0",
    "serde-1.0.136",
    "ripgrep-13.0.0",
    "regex-1.5.5",
    "clap-3.1.6",
    "hyper-0.14.18",
];

/// The default set of crates for opt-dist to collect rustc profiles.
pub const RUSTC_PGO_CRATES: &[&str] = &[
    "externs",
    "ctfe-stress-5",
    "cargo-0.60.0",
    "token-stream-stress",
    "match-stress",
    "tuple-stress",
    "diesel-1.4.8",
    "bitmaps-3.1.0",
];

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct JsonRoot {
    #[serde(default)] // For version 0 the field was not present.
    pub format_version: usize,
    pub system_stats: JsonInvocationSystemStats,
    pub invocations: Vec<JsonInvocation>,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct JsonInvocation {
    // Unix timestamp in seconds
    //
    // This is necessary to easily correlate this invocation with logs or other data.
    pub start_time: u64,
    pub duration_including_children_sec: f64,
    pub children: Vec<JsonNode>,
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum JsonNode {
    RustbuildStep {
        #[serde(rename = "type")]
        type_: String,
        debug_repr: String,

        duration_excluding_children_sec: f64,
        system_stats: JsonStepSystemStats,

        children: Vec<JsonNode>,
    },
    TestSuite(TestSuite),
}

#[derive(Serialize, Deserialize)]
pub struct TestSuite {
    pub metadata: TestSuiteMetadata,
    pub tests: Vec<Test>,
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TestSuiteMetadata {
    CargoPackage {
        crates: Vec<String>,
        target: String,
        host: String,
        stage: u32,
    },
    Compiletest {
        suite: String,
        mode: String,
        compare_mode: Option<String>,
        target: String,
        host: String,
        stage: u32,
    },
}

#[derive(Serialize, Deserialize)]
pub struct Test {
    pub name: String,
    #[serde(flatten)]
    pub outcome: TestOutcome,
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum TestOutcome {
    Passed,
    Failed,
    Ignored { ignore_reason: Option<String> },
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct JsonInvocationSystemStats {
    pub cpu_threads_count: usize,
    pub cpu_model: String,

    pub memory_total_bytes: u64,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct JsonStepSystemStats {
    pub cpu_utilization_percent: f64,
}
