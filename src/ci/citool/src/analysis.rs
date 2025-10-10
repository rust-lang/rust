use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::Debug;
use std::time::Duration;

use build_helper::metrics::{
    BuildStep, JsonRoot, TestOutcome, TestSuite, TestSuiteMetadata, escape_step_name,
    format_build_steps,
};

use crate::github::JobInfoResolver;
use crate::metrics::{JobMetrics, JobName, get_test_suites};
use crate::utils::{output_details, pluralize};
use crate::{metrics, utils};

/// Outputs durations of individual bootstrap steps from the gathered bootstrap invocations,
/// and also a table with summarized information about executed tests.
pub fn output_bootstrap_stats(metrics: &JsonRoot, parent_metrics: Option<&JsonRoot>) {
    if !metrics.invocations.is_empty() {
        println!("# Bootstrap steps");
        record_bootstrap_step_durations(&metrics, parent_metrics);
        record_test_suites(&metrics);
    }
}

fn record_bootstrap_step_durations(metrics: &JsonRoot, parent_metrics: Option<&JsonRoot>) {
    let parent_steps: HashMap<String, BuildStep> = parent_metrics
        .map(|metrics| {
            metrics
                .invocations
                .iter()
                .map(|invocation| {
                    (invocation.cmdline.clone(), BuildStep::from_invocation(invocation))
                })
                .collect()
        })
        .unwrap_or_default();

    for invocation in &metrics.invocations {
        let step = BuildStep::from_invocation(invocation);
        let table = format_build_steps(&step);
        eprintln!("Step `{}`\n{table}\n", invocation.cmdline);
        output_details(&format!("{} (steps)", invocation.cmdline), || {
            println!("<pre><code>{table}</code></pre>");
        });

        // If there was a parent bootstrap invocation with the same cmdline, diff it
        if let Some(parent_step) = parent_steps.get(&invocation.cmdline) {
            let table = format_build_step_diffs(&step, parent_step);

            let duration_before = parent_step.duration.as_secs();
            let duration_after = step.duration.as_secs();
            output_details(
                &format!("{} (diff) ({duration_before}s -> {duration_after}s)", invocation.cmdline),
                || {
                    println!("{table}");
                },
            );
        }
    }
    eprintln!("Recorded {} bootstrap invocation(s)", metrics.invocations.len());
}

/// Creates a table that displays a diff between the durations of steps between
/// two bootstrap invocations.
/// It also shows steps that were missing before/after.
fn format_build_step_diffs(current: &BuildStep, parent: &BuildStep) -> String {
    use std::fmt::Write;

    // Helper struct that compares steps by their full name
    struct StepByName<'a>((u32, &'a BuildStep));

    impl<'a> PartialEq for StepByName<'a> {
        fn eq(&self, other: &Self) -> bool {
            self.0.1.full_name.eq(&other.0.1.full_name)
        }
    }

    fn get_steps(step: &BuildStep) -> Vec<StepByName<'_>> {
        step.linearize_steps().into_iter().map(|v| StepByName(v)).collect()
    }

    let steps_before = get_steps(parent);
    let steps_after = get_steps(current);

    let mut table = "| Step | Before | After | Change |\n".to_string();
    writeln!(table, "|:-----|-------:|------:|-------:|").unwrap();

    // Try to match removed, added and same steps using a classic diff algorithm
    for result in diff::slice(&steps_before, &steps_after) {
        let (step, before, after, change) = match result {
            // The step was found both before and after
            diff::Result::Both(before, after) => {
                let duration_before = before.0.1.duration;
                let duration_after = after.0.1.duration;
                let pct_change = duration_after.as_secs_f64() / duration_before.as_secs_f64();
                let pct_change = pct_change * 100.0;
                // Normalize around 100, to get + for regression and - for improvements
                let pct_change = pct_change - 100.0;
                (
                    before,
                    format!("{:.2}s", duration_before.as_secs_f64()),
                    format!("{:.2}s", duration_after.as_secs_f64()),
                    format!("{pct_change:.1}%"),
                )
            }
            // The step was only found in the parent, so it was likely removed
            diff::Result::Left(removed) => (
                removed,
                format!("{:.2}s", removed.0.1.duration.as_secs_f64()),
                "".to_string(),
                "(removed)".to_string(),
            ),
            // The step was only found in the current commit, so it was likely added
            diff::Result::Right(added) => (
                added,
                "".to_string(),
                format!("{:.2}s", added.0.1.duration.as_secs_f64()),
                "(added)".to_string(),
            ),
        };

        let prefix = ".".repeat(step.0.0 as usize);
        writeln!(
            table,
            "| {prefix}{} | {before} | {after} | {change} |",
            escape_step_name(step.0.1),
        )
        .unwrap();
    }

    table
}

fn record_test_suites(metrics: &JsonRoot) {
    let suites = metrics::get_test_suites(&metrics);

    if !suites.is_empty() {
        let aggregated = aggregate_test_suites(&suites);
        let table = render_table(aggregated);
        println!("\n# Test results\n");
        println!("{table}");
    } else {
        eprintln!("No test suites found in metrics");
    }
}

fn render_table(suites: BTreeMap<String, TestSuiteRecord>) -> String {
    use std::fmt::Write;

    let mut table = "| Test suite | Passed ✅ | Ignored 🚫 | Failed  ❌ |\n".to_string();
    writeln!(table, "|:------|------:|------:|------:|").unwrap();

    fn compute_pct(value: f64, total: f64) -> f64 {
        if total == 0.0 { 0.0 } else { value / total }
    }

    fn write_row(
        buffer: &mut String,
        name: &str,
        record: &TestSuiteRecord,
        surround: &str,
    ) -> std::fmt::Result {
        let TestSuiteRecord { passed, ignored, failed } = record;
        let total = (record.passed + record.ignored + record.failed) as f64;
        let passed_pct = compute_pct(*passed as f64, total) * 100.0;
        let ignored_pct = compute_pct(*ignored as f64, total) * 100.0;
        let failed_pct = compute_pct(*failed as f64, total) * 100.0;

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

/// Outputs a report of test differences between the `parent` and `current` commits.
pub fn output_test_diffs(
    job_metrics: &HashMap<JobName, JobMetrics>,
    job_info_resolver: &mut JobInfoResolver,
) {
    let aggregated_test_diffs = aggregate_test_diffs(&job_metrics);
    report_test_diffs(aggregated_test_diffs, job_metrics, job_info_resolver);
}

/// Prints the ten largest differences in bootstrap durations.
pub fn output_largest_duration_changes(
    job_metrics: &HashMap<JobName, JobMetrics>,
    job_info_resolver: &mut JobInfoResolver,
) {
    struct Entry<'a> {
        job: &'a JobName,
        before: Duration,
        after: Duration,
        change: f64,
    }

    let mut changes: Vec<Entry> = vec![];
    for (job, metrics) in job_metrics {
        if let Some(parent) = &metrics.parent {
            let duration_before = parent
                .invocations
                .iter()
                .map(|i| BuildStep::from_invocation(i).duration)
                .sum::<Duration>();
            let duration_after = metrics
                .current
                .invocations
                .iter()
                .map(|i| BuildStep::from_invocation(i).duration)
                .sum::<Duration>();
            let pct_change = duration_after.as_secs_f64() / duration_before.as_secs_f64();
            let pct_change = pct_change * 100.0;
            // Normalize around 100, to get + for regression and - for improvements
            let pct_change = pct_change - 100.0;
            changes.push(Entry {
                job,
                before: duration_before,
                after: duration_after,
                change: pct_change,
            });
        }
    }
    changes.sort_by(|e1, e2| e1.change.abs().partial_cmp(&e2.change.abs()).unwrap().reverse());

    println!("# Job duration changes");
    for (index, entry) in changes.into_iter().take(10).enumerate() {
        println!(
            "{}. {}: {:.1}s -> {:.1}s ({:.1}%)",
            index + 1,
            format_job_link(job_info_resolver, job_metrics, entry.job),
            entry.before.as_secs_f64(),
            entry.after.as_secs_f64(),
            entry.change
        );
    }

    println!();
    output_details("How to interpret the job duration changes?", || {
        println!(
            r#"Job durations can vary a lot, based on the actual runner instance
that executed the job, system noise, invalidated caches, etc. The table above is provided
mostly for t-infra members, for simpler debugging of potential CI slow-downs."#
        );
    });
}

#[derive(Default)]
struct TestSuiteRecord {
    passed: u64,
    ignored: u64,
    failed: u64,
}

fn test_metadata_name(metadata: &TestSuiteMetadata) -> String {
    match metadata {
        TestSuiteMetadata::CargoPackage { crates, stage, .. } => {
            format!("{} (stage {stage})", crates.join(", "))
        }
        TestSuiteMetadata::Compiletest { suite, stage, .. } => {
            format!("{suite} (stage {stage})")
        }
    }
}

fn aggregate_test_suites(suites: &[&TestSuite]) -> BTreeMap<String, TestSuiteRecord> {
    let mut records: BTreeMap<String, TestSuiteRecord> = BTreeMap::new();
    for suite in suites {
        let name = test_metadata_name(&suite.metadata);
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

/// Represents a difference in the outcome of tests between a base and a current commit.
/// Maps test diffs to jobs that contained them.
#[derive(Debug)]
struct AggregatedTestDiffs {
    diffs: HashMap<TestDiff, Vec<JobName>>,
}

fn aggregate_test_diffs(jobs: &HashMap<JobName, JobMetrics>) -> AggregatedTestDiffs {
    let mut diffs: HashMap<TestDiff, Vec<JobName>> = HashMap::new();

    // Aggregate test suites
    for (name, metrics) in jobs {
        if let Some(parent) = &metrics.parent {
            let tests_parent = aggregate_tests(parent);
            let tests_current = aggregate_tests(&metrics.current);
            for diff in calculate_test_diffs(tests_parent, tests_current) {
                diffs.entry(diff).or_default().push(name.to_string());
            }
        }
    }

    AggregatedTestDiffs { diffs }
}

#[derive(Eq, PartialEq, Hash, Debug)]
enum TestOutcomeDiff {
    ChangeOutcome { before: TestOutcome, after: TestOutcome },
    Missing { before: TestOutcome },
    Added(TestOutcome),
}

#[derive(Eq, PartialEq, Hash, Debug)]
struct TestDiff {
    test: Test,
    diff: TestOutcomeDiff,
}

fn calculate_test_diffs(parent: TestSuiteData, current: TestSuiteData) -> HashSet<TestDiff> {
    let mut diffs = HashSet::new();
    for (test, outcome) in &current.tests {
        match parent.tests.get(test) {
            Some(before) => {
                if before != outcome {
                    diffs.insert(TestDiff {
                        test: test.clone(),
                        diff: TestOutcomeDiff::ChangeOutcome {
                            before: before.clone(),
                            after: outcome.clone(),
                        },
                    });
                }
            }
            None => {
                diffs.insert(TestDiff {
                    test: test.clone(),
                    diff: TestOutcomeDiff::Added(outcome.clone()),
                });
            }
        }
    }
    for (test, outcome) in &parent.tests {
        if !current.tests.contains_key(test) {
            diffs.insert(TestDiff {
                test: test.clone(),
                diff: TestOutcomeDiff::Missing { before: outcome.clone() },
            });
        }
    }

    diffs
}

/// Aggregates test suite executions from all bootstrap invocations in a given CI job.
#[derive(Default)]
struct TestSuiteData {
    tests: HashMap<Test, TestOutcome>,
}

#[derive(Hash, PartialEq, Eq, Debug, Clone)]
struct Test {
    name: String,
    stage: u8,
    is_doctest: bool,
}

/// Extracts all tests from the passed metrics and map them to their outcomes.
fn aggregate_tests(metrics: &JsonRoot) -> TestSuiteData {
    let mut tests = HashMap::new();
    let test_suites = get_test_suites(&metrics);
    for suite in test_suites {
        let stage = match suite.metadata {
            TestSuiteMetadata::CargoPackage { stage, .. } => stage,
            TestSuiteMetadata::Compiletest { stage, .. } => stage,
        } as u8;
        for test in &suite.tests {
            // Poor man's detection of doctests based on the "(line XYZ)" suffix
            let is_doctest = matches!(suite.metadata, TestSuiteMetadata::CargoPackage { .. })
                && test.name.contains("(line");
            let test_entry = Test {
                name: utils::normalize_path_delimiters(&test.name).to_string(),
                stage,
                is_doctest,
            };
            tests.insert(test_entry, test.outcome.clone());
        }
    }
    TestSuiteData { tests }
}

/// Prints test changes in Markdown format to stdout.
fn report_test_diffs(
    diff: AggregatedTestDiffs,
    job_metrics: &HashMap<JobName, JobMetrics>,
    job_info_resolver: &mut JobInfoResolver,
) {
    println!("# Test differences");
    if diff.diffs.is_empty() {
        println!("No test diffs found");
        return;
    }

    fn format_outcome(outcome: &TestOutcome) -> String {
        match outcome {
            TestOutcome::Passed => "pass".to_string(),
            TestOutcome::Failed => "fail".to_string(),
            TestOutcome::Ignored { ignore_reason } => {
                let reason = match ignore_reason {
                    Some(reason) => format!(" ({reason})"),
                    None => String::new(),
                };
                format!("ignore{reason}")
            }
        }
    }

    fn format_diff(diff: &TestOutcomeDiff) -> String {
        match diff {
            TestOutcomeDiff::ChangeOutcome { before, after } => {
                format!("{} -> {}", format_outcome(before), format_outcome(after))
            }
            TestOutcomeDiff::Missing { before } => {
                format!("{} -> [missing]", format_outcome(before))
            }
            TestOutcomeDiff::Added(outcome) => {
                format!("[missing] -> {}", format_outcome(outcome))
            }
        }
    }

    fn format_job_group(group: u64) -> String {
        format!("**J{group}**")
    }

    // It would be quite noisy to repeat the jobs that contained the test changes after/next to
    // every test diff. At the same time, grouping the test diffs by
    // [unique set of jobs that contained them] also doesn't work well, because the test diffs
    // would have to be duplicated several times.
    // Instead, we create a set of unique job groups, and then print a job group after each test.
    // We then print the job groups at the end, as a sort of index.
    let mut grouped_diffs: Vec<(&TestDiff, u64)> = vec![];
    let mut job_list_to_group: HashMap<&[JobName], u64> = HashMap::new();
    let mut job_index: Vec<&[JobName]> = vec![];

    let original_diff_count = diff.diffs.len();
    let diffs = diff
        .diffs
        .into_iter()
        .filter(|(diff, _)| !diff.test.is_doctest)
        .map(|(diff, mut jobs)| {
            jobs.sort();
            (diff, jobs)
        })
        .collect::<Vec<_>>();
    let doctest_count = original_diff_count.saturating_sub(diffs.len());

    let max_diff_count = 100;
    for (diff, jobs) in diffs.iter().take(max_diff_count) {
        let jobs = &*jobs;
        let job_group = match job_list_to_group.get(jobs.as_slice()) {
            Some(id) => *id,
            None => {
                let id = job_index.len() as u64;
                job_index.push(jobs);
                job_list_to_group.insert(jobs, id);
                id
            }
        };
        grouped_diffs.push((diff, job_group));
    }

    // Sort diffs by job group and test name
    grouped_diffs.sort_by(|(d1, g1), (d2, g2)| g1.cmp(&g2).then(d1.test.name.cmp(&d2.test.name)));

    // Now group the tests by stage
    let mut grouped_by_stage: BTreeMap<u8, Vec<(&TestDiff, u64)>> = Default::default();
    for (diff, group) in grouped_diffs {
        grouped_by_stage.entry(diff.test.stage).or_default().push((diff, group))
    }

    output_details(
        &format!("Show {} test {}\n", original_diff_count, pluralize("diff", original_diff_count)),
        || {
            for (stage, diffs) in grouped_by_stage {
                println!("## Stage {stage}");
                for (diff, job_group) in diffs {
                    println!(
                        "- `{}`: {} ({})",
                        diff.test.name,
                        format_diff(&diff.diff),
                        format_job_group(job_group)
                    );
                }
            }

            let extra_diffs = diffs.len().saturating_sub(max_diff_count);
            if extra_diffs > 0 {
                println!(
                    "\n(and {extra_diffs} additional {})",
                    pluralize("test diff", extra_diffs)
                );
            }

            if doctest_count > 0 {
                let prefix =
                    if doctest_count < original_diff_count { "Additionally, " } else { "" };
                println!(
                    "\n{prefix}{doctest_count} doctest {} were found. These are ignored, as they are noisy.",
                    pluralize("diff", doctest_count)
                );
            }

            // Now print the job group index
            if !job_index.is_empty() {
                println!("\n**Job group index**\n");
                for (group, jobs) in job_index.into_iter().enumerate() {
                    println!(
                        "- {}: {}",
                        format_job_group(group as u64),
                        jobs.iter()
                            .map(|j| format_job_link(job_info_resolver, job_metrics, j))
                            .collect::<Vec<_>>()
                            .join(", ")
                    );
                }
            }
        },
    );
}

/// Tries to get a GitHub Actions job summary URL from the resolver.
/// If it is not available, just wraps the job name in backticks.
fn format_job_link(
    job_info_resolver: &mut JobInfoResolver,
    job_metrics: &HashMap<JobName, JobMetrics>,
    job_name: &str,
) -> String {
    job_metrics
        .get(job_name)
        .and_then(|metrics| job_info_resolver.get_job_summary_link(job_name, &metrics.current))
        .map(|summary_url| format!("[{job_name}]({summary_url})"))
        .unwrap_or_else(|| format!("`{job_name}`"))
}
