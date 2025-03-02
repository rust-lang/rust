use std::collections::BTreeMap;
use std::fs::File;
use std::io::Write;
use std::path::Path;

use anyhow::Context;
use build_helper::metrics::{
    BuildStep, JsonNode, JsonRoot, TestOutcome, TestSuite, TestSuiteMetadata, format_build_steps,
};

pub fn postprocess_metrics(metrics_path: &Path, summary_path: &Path) -> anyhow::Result<()> {
    let metrics = load_metrics(metrics_path)?;

    let mut file = File::options()
        .append(true)
        .create(true)
        .open(summary_path)
        .with_context(|| format!("Cannot open summary file at {summary_path:?}"))?;

    if !metrics.invocations.is_empty() {
        writeln!(file, "# Bootstrap steps")?;
        record_bootstrap_step_durations(&metrics, &mut file)?;
        record_test_suites(&metrics, &mut file)?;
    }

    Ok(())
}

fn record_bootstrap_step_durations(metrics: &JsonRoot, file: &mut File) -> anyhow::Result<()> {
    for invocation in &metrics.invocations {
        let step = BuildStep::from_invocation(invocation);
        let table = format_build_steps(&step);
        eprintln!("Step `{}`\n{table}\n", invocation.cmdline);
        writeln!(
            file,
            r"<details>
<summary>{}</summary>
<pre><code>{table}</code></pre>
</details>
",
            invocation.cmdline
        )?;
    }
    eprintln!("Recorded {} bootstrap invocation(s)", metrics.invocations.len());

    Ok(())
}

fn record_test_suites(metrics: &JsonRoot, file: &mut File) -> anyhow::Result<()> {
    let suites = get_test_suites(&metrics);

    if !suites.is_empty() {
        let aggregated = aggregate_test_suites(&suites);
        let table = render_table(aggregated);
        writeln!(file, "\n# Test results\n")?;
        writeln!(file, "{table}")?;
    } else {
        eprintln!("No test suites found in metrics");
    }

    Ok(())
}

fn render_table(suites: BTreeMap<String, TestSuiteRecord>) -> String {
    use std::fmt::Write;

    let mut table = "| Test suite | Passed ✅ | Ignored 🚫 | Failed  ❌ |\n".to_string();
    writeln!(table, "|:------|------:|------:|------:|").unwrap();

    fn write_row(
        buffer: &mut String,
        name: &str,
        record: &TestSuiteRecord,
        surround: &str,
    ) -> std::fmt::Result {
        let TestSuiteRecord { passed, ignored, failed } = record;
        let total = (record.passed + record.ignored + record.failed) as f64;
        let passed_pct = ((*passed as f64) / total) * 100.0;
        let ignored_pct = ((*ignored as f64) / total) * 100.0;
        let failed_pct = ((*failed as f64) / total) * 100.0;

        write!(buffer, "| {surround}{name}{surround} |")?;
        write!(buffer, " {surround}{passed} ({passed_pct:.0}%){surround} |")?;
        write!(buffer, " {surround}{ignored} ({ignored_pct:.0}%){surround} |")?;
        writeln!(buffer, " {surround}{failed} ({failed_pct:.0}%){surround} |")?;

        Ok(())
    }

    let mut total = TestSuiteRecord::default();
    for (name, record) in suites {
        write_row(&mut table, &name, &record, "").unwrap();
        total.passed += record.passed;
        total.ignored += record.ignored;
        total.failed += record.failed;
    }
    write_row(&mut table, "Total", &total, "**").unwrap();
    table
}

#[derive(Default)]
struct TestSuiteRecord {
    passed: u64,
    ignored: u64,
    failed: u64,
}

fn aggregate_test_suites(suites: &[&TestSuite]) -> BTreeMap<String, TestSuiteRecord> {
    let mut records: BTreeMap<String, TestSuiteRecord> = BTreeMap::new();
    for suite in suites {
        let name = match &suite.metadata {
            TestSuiteMetadata::CargoPackage { crates, stage, .. } => {
                format!("{} (stage {stage})", crates.join(", "))
            }
            TestSuiteMetadata::Compiletest { suite, stage, .. } => {
                format!("{suite} (stage {stage})")
            }
        };
        let record = records.entry(name).or_default();
        for test in &suite.tests {
            match test.outcome {
                TestOutcome::Passed => {
                    record.passed += 1;
                }
                TestOutcome::Failed => {
                    record.failed += 1;
                }
                TestOutcome::Ignored { .. } => {
                    record.ignored += 1;
                }
            }
        }
    }
    records
}

fn get_test_suites(metrics: &JsonRoot) -> Vec<&TestSuite> {
    fn visit_test_suites<'a>(nodes: &'a [JsonNode], suites: &mut Vec<&'a TestSuite>) {
        for node in nodes {
            match node {
                JsonNode::RustbuildStep { children, .. } => {
                    visit_test_suites(&children, suites);
                }
                JsonNode::TestSuite(suite) => {
                    suites.push(&suite);
                }
            }
        }
    }

    let mut suites = vec![];
    for invocation in &metrics.invocations {
        visit_test_suites(&invocation.children, &mut suites);
    }
    suites
}

fn load_metrics(path: &Path) -> anyhow::Result<JsonRoot> {
    let metrics = std::fs::read_to_string(path)
        .with_context(|| format!("Cannot read JSON metrics from {path:?}"))?;
    let metrics: JsonRoot = serde_json::from_str(&metrics)
        .with_context(|| format!("Cannot deserialize JSON metrics from {path:?}"))?;
    Ok(metrics)
}
