use std::time::Duration;

use serde_derive::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct JsonRoot {
    #[serde(default)] // For version 0 the field was not present.
    pub format_version: usize,
    pub system_stats: JsonInvocationSystemStats,
    pub invocations: Vec<JsonInvocation>,
    #[serde(default)]
    pub ci_metadata: Option<CiMetadata>,
}

/// Represents metadata about bootstrap's execution in CI.
#[derive(Serialize, Deserialize)]
pub struct CiMetadata {
    /// GitHub run ID of the workflow where bootstrap was executed.
    /// Note that the run ID will be shared amongst all jobs executed in that workflow.
    pub workflow_run_id: u64,
    /// Full name of a GitHub repository where bootstrap was executed in CI.
    /// e.g. `rust-lang-ci/rust`.
    pub repository: String,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct JsonInvocation {
    // Remembers the command-line invocation with which bootstrap was invoked.
    pub cmdline: String,
    // Unix timestamp in seconds
    //
    // This is necessary to easily correlate this invocation with logs or other data.
    pub start_time: u64,
    #[serde(deserialize_with = "null_as_f64_nan")]
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

        #[serde(deserialize_with = "null_as_f64_nan")]
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

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Hash)]
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
    #[serde(deserialize_with = "null_as_f64_nan")]
    pub cpu_utilization_percent: f64,
}

fn null_as_f64_nan<'de, D: serde::Deserializer<'de>>(d: D) -> Result<f64, D::Error> {
    use serde::Deserialize as _;
    Option::<f64>::deserialize(d).map(|f| f.unwrap_or(f64::NAN))
}

/// Represents a single bootstrap step, with the accumulated duration of all its children.
#[derive(Clone, Debug)]
pub struct BuildStep {
    pub r#type: String,
    pub children: Vec<BuildStep>,
    pub duration: Duration,
    // Full name of the step, including all parent names
    pub full_name: String,
}

impl BuildStep {
    /// Create a `BuildStep` representing a single invocation of bootstrap.
    /// The most important thing is that the build step aggregates the
    /// durations of all children, so that it can be easily accessed.
    pub fn from_invocation(invocation: &JsonInvocation) -> Self {
        fn parse(node: &JsonNode, parent_name: &str) -> Option<BuildStep> {
            match node {
                JsonNode::RustbuildStep {
                    type_: kind,
                    children,
                    duration_excluding_children_sec,
                    ..
                } => {
                    let full_name = format!("{parent_name}-{kind}");
                    let children: Vec<_> =
                        children.iter().filter_map(|s| parse(s, &full_name)).collect();
                    let children_duration = children.iter().map(|c| c.duration).sum::<Duration>();
                    Some(BuildStep {
                        r#type: kind.to_string(),
                        children,
                        full_name,
                        duration: children_duration
                            + Duration::from_secs_f64(*duration_excluding_children_sec),
                    })
                }
                JsonNode::TestSuite(_) => None,
            }
        }

        let duration = Duration::from_secs_f64(invocation.duration_including_children_sec);

        // The root "total" step is kind of a virtual step that encompasses all other steps,
        // but it is not a real parent of the other steps.
        // We thus start the parent name hierarchy here and use an empty string
        // as the parent name of the top-level steps.
        let children: Vec<_> = invocation.children.iter().filter_map(|s| parse(s, "")).collect();
        Self { r#type: "total".to_string(), children, duration, full_name: "total".to_string() }
    }

    pub fn find_all_by_type(&self, r#type: &str) -> Vec<&Self> {
        let mut result = Vec::new();
        self.find_by_type(r#type, &mut result);
        result
    }

    fn find_by_type<'a>(&'a self, r#type: &str, result: &mut Vec<&'a Self>) {
        if self.r#type == r#type {
            result.push(self);
        }
        for child in &self.children {
            child.find_by_type(r#type, result);
        }
    }

    /// Returns a Vec with all substeps, ordered by their hierarchical order.
    /// The first element of the tuple is the depth of a given step.
    pub fn linearize_steps(&self) -> Vec<(u32, &BuildStep)> {
        let mut substeps: Vec<(u32, &BuildStep)> = Vec::new();

        fn visit<'a>(step: &'a BuildStep, level: u32, substeps: &mut Vec<(u32, &'a BuildStep)>) {
            substeps.push((level, step));
            for child in &step.children {
                visit(child, level + 1, substeps);
            }
        }

        visit(self, 0, &mut substeps);
        substeps
    }
}

/// Writes build steps into a nice indented table.
pub fn format_build_steps(root: &BuildStep) -> String {
    use std::fmt::Write;

    let mut output = String::new();
    for (level, step) in root.linearize_steps() {
        let label = format!("{}{}", ".".repeat(level as usize), escape_step_name(step));
        writeln!(output, "{label:.<65}{:>8.2}s", step.duration.as_secs_f64()).unwrap();
    }
    output
}

/// Bootstrap steps can be generic and thus contain angle brackets (<...>).
/// However, Markdown interprets these as HTML, so we need to escap ethem.
pub fn escape_step_name(step: &BuildStep) -> String {
    step.r#type.replace('<', "&lt;").replace('>', "&gt;")
}
