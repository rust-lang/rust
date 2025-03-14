use std::cmp::Reverse;
use std::collections::HashMap;

use anyhow::Context;
use build_helper::metrics::{JsonRoot, TestOutcome};

use crate::jobs::JobDatabase;
use crate::metrics::get_test_suites;

type Sha = String;
type JobName = String;

/// Computes a post merge CI analysis report between the `parent` and `current` commits.
pub fn post_merge_report(job_db: JobDatabase, parent: Sha, current: Sha) -> anyhow::Result<()> {
    let jobs = download_all_metrics(&job_db, &parent, &current)?;
    let diffs = aggregate_test_diffs(&jobs)?;
    report_test_changes(diffs);

    Ok(())
}

struct JobMetrics {
    parent: Option<JsonRoot>,
    current: JsonRoot,
}

/// Download before/after metrics for all auto jobs in the job database.
fn download_all_metrics(
    job_db: &JobDatabase,
    parent: &str,
    current: &str,
) -> anyhow::Result<HashMap<JobName, JobMetrics>> {
    let mut jobs = HashMap::default();

    for job in &job_db.auto_jobs {
        eprintln!("Downloading metrics of job {}", job.name);
        let metrics_parent = match download_job_metrics(&job.name, parent) {
            Ok(metrics) => Some(metrics),
            Err(error) => {
                eprintln!(
                    r#"Did not find metrics for job `{}` at `{}`: {error:?}.
Maybe it was newly added?"#,
                    job.name, parent
                );
                None
            }
        };
        let metrics_current = download_job_metrics(&job.name, current)?;
        jobs.insert(
            job.name.clone(),
            JobMetrics { parent: metrics_parent, current: metrics_current },
        );
    }
    Ok(jobs)
}

fn download_job_metrics(job_name: &str, sha: &str) -> anyhow::Result<JsonRoot> {
    let url = get_metrics_url(job_name, sha);
    let mut response = ureq::get(&url).call()?;
    if !response.status().is_success() {
        return Err(anyhow::anyhow!(
            "Cannot fetch metrics from {url}: {}\n{}",
            response.status(),
            response.body_mut().read_to_string()?
        ));
    }
    let data: JsonRoot = response
        .body_mut()
        .read_json()
        .with_context(|| anyhow::anyhow!("cannot deserialize metrics from {url}"))?;
    Ok(data)
}

fn get_metrics_url(job_name: &str, sha: &str) -> String {
    let suffix = if job_name.ends_with("-alt") { "-alt" } else { "" };
    format!("https://ci-artifacts.rust-lang.org/rustc-builds{suffix}/{sha}/metrics-{job_name}.json")
}

fn aggregate_test_diffs(
    jobs: &HashMap<JobName, JobMetrics>,
) -> anyhow::Result<Vec<AggregatedTestDiffs>> {
    let mut job_diffs = vec![];

    // Aggregate test suites
    for (name, metrics) in jobs {
        if let Some(parent) = &metrics.parent {
            let tests_parent = aggregate_tests(parent);
            let tests_current = aggregate_tests(&metrics.current);
            let test_diffs = calculate_test_diffs(tests_parent, tests_current);
            if !test_diffs.is_empty() {
                job_diffs.push((name.clone(), test_diffs));
            }
        }
    }

    // Aggregate jobs with the same diff, as often the same diff will appear in many jobs
    let job_diffs: HashMap<Vec<(Test, TestOutcomeDiff)>, Vec<String>> =
        job_diffs.into_iter().fold(HashMap::new(), |mut acc, (job, diffs)| {
            acc.entry(diffs).or_default().push(job);
            acc
        });

    Ok(job_diffs
        .into_iter()
        .map(|(test_diffs, jobs)| AggregatedTestDiffs { jobs, test_diffs })
        .collect())
}

fn calculate_test_diffs(
    reference: TestSuiteData,
    current: TestSuiteData,
) -> Vec<(Test, TestOutcomeDiff)> {
    let mut diffs = vec![];
    for (test, outcome) in &current.tests {
        match reference.tests.get(test) {
            Some(before) => {
                if before != outcome {
                    diffs.push((
                        test.clone(),
                        TestOutcomeDiff::ChangeOutcome {
                            before: before.clone(),
                            after: outcome.clone(),
                        },
                    ));
                }
            }
            None => diffs.push((test.clone(), TestOutcomeDiff::Added(outcome.clone()))),
        }
    }
    for (test, outcome) in &reference.tests {
        if !current.tests.contains_key(test) {
            diffs.push((test.clone(), TestOutcomeDiff::Missing { before: outcome.clone() }));
        }
    }

    diffs
}

/// Represents a difference in the outcome of tests between a base and a current commit.
#[derive(Debug)]
struct AggregatedTestDiffs {
    /// All jobs that had the exact same test diffs.
    jobs: Vec<String>,
    test_diffs: Vec<(Test, TestOutcomeDiff)>,
}

#[derive(Eq, PartialEq, Hash, Debug)]
enum TestOutcomeDiff {
    ChangeOutcome { before: TestOutcome, after: TestOutcome },
    Missing { before: TestOutcome },
    Added(TestOutcome),
}

/// Aggregates test suite executions from all bootstrap invocations in a given CI job.
#[derive(Default)]
struct TestSuiteData {
    tests: HashMap<Test, TestOutcome>,
}

#[derive(Hash, PartialEq, Eq, Debug, Clone)]
struct Test {
    name: String,
}

/// Extracts all tests from the passed metrics and map them to their outcomes.
fn aggregate_tests(metrics: &JsonRoot) -> TestSuiteData {
    let mut tests = HashMap::new();
    let test_suites = get_test_suites(&metrics);
    for suite in test_suites {
        for test in &suite.tests {
            let test_entry = Test { name: normalize_test_name(&test.name) };
            tests.insert(test_entry, test.outcome.clone());
        }
    }
    TestSuiteData { tests }
}

/// Normalizes Windows-style path delimiters to Unix-style paths.
fn normalize_test_name(name: &str) -> String {
    name.replace('\\', "/")
}

/// Prints test changes in Markdown format to stdout.
fn report_test_changes(mut diffs: Vec<AggregatedTestDiffs>) {
    println!("## Test differences");
    if diffs.is_empty() {
        println!("No test diffs found");
        return;
    }

    // Sort diffs in decreasing order by diff count
    diffs.sort_by_key(|entry| Reverse(entry.test_diffs.len()));

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

    let max_diff_count = 10;
    let max_job_count = 5;
    let max_test_count = 10;

    for diff in diffs.iter().take(max_diff_count) {
        let mut jobs = diff.jobs.clone();
        jobs.sort();

        let jobs = jobs.iter().take(max_job_count).map(|j| format!("`{j}`")).collect::<Vec<_>>();

        let extra_jobs = diff.jobs.len().saturating_sub(max_job_count);
        let suffix = if extra_jobs > 0 {
            format!(" (and {extra_jobs} {})", pluralize("other", extra_jobs))
        } else {
            String::new()
        };
        println!("- {}{suffix}", jobs.join(","));

        let extra_tests = diff.test_diffs.len().saturating_sub(max_test_count);
        for (test, outcome_diff) in diff.test_diffs.iter().take(max_test_count) {
            println!("  - {}: {}", test.name, format_diff(&outcome_diff));
        }
        if extra_tests > 0 {
            println!("  - (and {extra_tests} additional {})", pluralize("tests", extra_tests));
        }
    }

    let extra_diffs = diffs.len().saturating_sub(max_diff_count);
    if extra_diffs > 0 {
        println!("\n(and {extra_diffs} additional {})", pluralize("diff", extra_diffs));
    }
}

fn pluralize(text: &str, count: usize) -> String {
    if count == 1 { text.to_string() } else { format!("{text}s") }
}
