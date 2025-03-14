use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use anyhow::Context;
use build_helper::metrics::{JsonRoot, TestOutcome, TestSuiteMetadata};

use crate::jobs::JobDatabase;
use crate::metrics::get_test_suites;

type Sha = String;
type JobName = String;

/// Computes a post merge CI analysis report between the `parent` and `current` commits.
pub fn post_merge_report(job_db: JobDatabase, parent: Sha, current: Sha) -> anyhow::Result<()> {
    let jobs = download_all_metrics(&job_db, &parent, &current)?;
    let aggregated_test_diffs = aggregate_test_diffs(&jobs)?;

    println!("Comparing {parent} (base) -> {current} (this PR)\n");
    report_test_diffs(aggregated_test_diffs);

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

/// Downloads job metrics of the given job for the given commit.
/// Caches the result on the local disk.
fn download_job_metrics(job_name: &str, sha: &str) -> anyhow::Result<JsonRoot> {
    let cache_path = PathBuf::from(".citool-cache").join(sha).join(job_name).join("metrics.json");
    if let Some(cache_entry) =
        std::fs::read_to_string(&cache_path).ok().and_then(|data| serde_json::from_str(&data).ok())
    {
        return Ok(cache_entry);
    }

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

    // Ignore errors if cache cannot be created
    if std::fs::create_dir_all(cache_path.parent().unwrap()).is_ok() {
        if let Ok(serialized) = serde_json::to_string(&data) {
            let _ = std::fs::write(&cache_path, &serialized);
        }
    }
    Ok(data)
}

fn get_metrics_url(job_name: &str, sha: &str) -> String {
    let suffix = if job_name.ends_with("-alt") { "-alt" } else { "" };
    format!("https://ci-artifacts.rust-lang.org/rustc-builds{suffix}/{sha}/metrics-{job_name}.json")
}

/// Represents a difference in the outcome of tests between a base and a current commit.
/// Maps test diffs to jobs that contained them.
#[derive(Debug)]
struct AggregatedTestDiffs {
    diffs: HashMap<TestDiff, Vec<JobName>>,
}

fn aggregate_test_diffs(
    jobs: &HashMap<JobName, JobMetrics>,
) -> anyhow::Result<AggregatedTestDiffs> {
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

    Ok(AggregatedTestDiffs { diffs })
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
    is_doctest: bool,
}

/// Extracts all tests from the passed metrics and map them to their outcomes.
fn aggregate_tests(metrics: &JsonRoot) -> TestSuiteData {
    let mut tests = HashMap::new();
    let test_suites = get_test_suites(&metrics);
    for suite in test_suites {
        for test in &suite.tests {
            // Poor man's detection of doctests based on the "(line XYZ)" suffix
            let is_doctest = matches!(suite.metadata, TestSuiteMetadata::CargoPackage { .. })
                && test.name.contains("(line");
            let test_entry = Test { name: normalize_test_name(&test.name), is_doctest };
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
fn report_test_diffs(diff: AggregatedTestDiffs) {
    println!("## Test differences");
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

    for (diff, job_group) in grouped_diffs {
        println!(
            "- `{}`: {} ({})",
            diff.test.name,
            format_diff(&diff.diff),
            format_job_group(job_group)
        );
    }

    let extra_diffs = diffs.len().saturating_sub(max_diff_count);
    if extra_diffs > 0 {
        println!("\n(and {extra_diffs} additional {})", pluralize("test diff", extra_diffs));
    }

    if doctest_count > 0 {
        println!(
            "\nAdditionally, {doctest_count} doctest {} were found. These are ignored, as they are noisy.",
            pluralize("diff", doctest_count)
        );
    }

    // Now print the job group index
    println!("\n**Job group index**\n");
    for (group, jobs) in job_index.into_iter().enumerate() {
        println!(
            "- {}: {}",
            format_job_group(group as u64),
            jobs.iter().map(|j| format!("`{j}`")).collect::<Vec<_>>().join(", ")
        );
    }
}

fn pluralize(text: &str, count: usize) -> String {
    if count == 1 { text.to_string() } else { format!("{text}s") }
}
