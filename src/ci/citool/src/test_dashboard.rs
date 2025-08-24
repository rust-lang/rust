use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};

use askama::Template;
use build_helper::metrics::{TestOutcome, TestSuiteMetadata};

use crate::jobs::JobDatabase;
use crate::metrics::{JobMetrics, JobName, download_auto_job_metrics, get_test_suites};
use crate::utils::normalize_path_delimiters;

/// Generate a set of HTML files into a directory that contain a dashboard of test results.
pub fn generate_test_dashboard(
    db: JobDatabase,
    current: &str,
    output_dir: &Path,
) -> anyhow::Result<()> {
    let metrics = download_auto_job_metrics(&db, None, current)?;
    let suites = gather_test_suites(&metrics);

    std::fs::create_dir_all(output_dir)?;

    let test_count = suites.test_count();
    write_page(output_dir, "index.html", &TestSuitesPage { suites, test_count })?;

    Ok(())
}

fn write_page<T: Template>(dir: &Path, name: &str, template: &T) -> anyhow::Result<()> {
    let mut file = BufWriter::new(File::create(dir.join(name))?);
    Template::write_into(template, &mut file)?;
    Ok(())
}

fn gather_test_suites(job_metrics: &HashMap<JobName, JobMetrics>) -> TestSuites<'_> {
    struct CoarseTestSuite<'a> {
        tests: BTreeMap<String, Test<'a>>,
    }

    let mut suites: HashMap<String, CoarseTestSuite> = HashMap::new();

    // First, gather tests from all jobs, stages and targets, and aggregate them per suite
    // Only work with compiletest suites.
    for (job, metrics) in job_metrics {
        let test_suites = get_test_suites(&metrics.current);
        for suite in test_suites {
            let (suite_name, stage, target) = match &suite.metadata {
                TestSuiteMetadata::CargoPackage { .. } => {
                    continue;
                }
                TestSuiteMetadata::Compiletest { suite, stage, target, .. } => {
                    (suite.clone(), *stage, target)
                }
            };
            let suite_entry = suites
                .entry(suite_name.clone())
                .or_insert_with(|| CoarseTestSuite { tests: Default::default() });
            let test_metadata = TestMetadata { job, stage, target };

            for test in &suite.tests {
                let test_name = normalize_test_name(&test.name, &suite_name);
                let (test_name, variant_name) = match test_name.rsplit_once('#') {
                    Some((name, variant)) => (name.to_string(), variant.to_string()),
                    None => (test_name, "".to_string()),
                };
                let test_entry = suite_entry
                    .tests
                    .entry(test_name.clone())
                    .or_insert_with(|| Test { revisions: Default::default() });
                let variant_entry = test_entry
                    .revisions
                    .entry(variant_name)
                    .or_insert_with(|| TestResults { passed: vec![], ignored: vec![] });

                match test.outcome {
                    TestOutcome::Passed => {
                        variant_entry.passed.push(test_metadata);
                    }
                    TestOutcome::Ignored { ignore_reason: _ } => {
                        variant_entry.ignored.push(test_metadata);
                    }
                    TestOutcome::Failed => {
                        eprintln!("Warning: failed test {test_name}");
                    }
                }
            }
        }
    }

    // Then, split the suites per directory
    let mut suites = suites.into_iter().collect::<Vec<_>>();
    suites.sort_by(|a, b| a.0.cmp(&b.0));

    let suites = suites
        .into_iter()
        .map(|(suite_name, suite)| TestSuite { group: build_test_group(&suite_name, suite.tests) })
        .collect();

    TestSuites { suites }
}

/// Recursively expand a test group based on filesystem hierarchy.
fn build_test_group<'a>(name: &str, tests: BTreeMap<String, Test<'a>>) -> TestGroup<'a> {
    let mut root_tests = vec![];
    let mut subdirs: BTreeMap<String, BTreeMap<String, Test<'a>>> = Default::default();

    // Split tests into root tests and tests located in subdirectories
    for (name, test) in tests {
        let mut components = Path::new(&name).components().peekable();
        let subdir = components.next().unwrap();

        if components.peek().is_none() {
            // This is a root test
            root_tests.push((name, test));
        } else {
            // This is a test in a nested directory
            let subdir_tests =
                subdirs.entry(subdir.as_os_str().to_str().unwrap().to_string()).or_default();
            let test_name =
                components.into_iter().collect::<PathBuf>().to_str().unwrap().to_string();
            subdir_tests.insert(test_name, test);
        }
    }
    let dirs = subdirs
        .into_iter()
        .map(|(name, tests)| {
            let group = build_test_group(&name, tests);
            (name, group)
        })
        .collect();

    TestGroup { name: name.to_string(), root_tests, groups: dirs }
}

/// Compiletest tests start with `[suite] tests/[suite]/a/b/c...`.
/// Remove the `[suite] tests/[suite]/` prefix so that we can find the filesystem path.
/// Also normalizes path delimiters.
fn normalize_test_name(name: &str, suite_name: &str) -> String {
    let name = normalize_path_delimiters(name);
    let name = name.as_ref();
    let name = name.strip_prefix(&format!("[{suite_name}]")).unwrap_or(name).trim();
    let name = name.strip_prefix("tests/").unwrap_or(name);
    let name = name.strip_prefix(suite_name).unwrap_or(name);
    name.trim_start_matches("/").to_string()
}

struct TestSuites<'a> {
    suites: Vec<TestSuite<'a>>,
}

impl<'a> TestSuites<'a> {
    fn test_count(&self) -> u64 {
        self.suites.iter().map(|suite| suite.group.test_count()).sum::<u64>()
    }
}

struct TestSuite<'a> {
    group: TestGroup<'a>,
}

struct TestResults<'a> {
    passed: Vec<TestMetadata<'a>>,
    ignored: Vec<TestMetadata<'a>>,
}

struct Test<'a> {
    revisions: BTreeMap<String, TestResults<'a>>,
}

impl<'a> Test<'a> {
    /// If this is a test without revisions, it will have a single entry in `revisions` with
    /// an empty string as the revision name.
    fn single_test(&self) -> Option<&TestResults<'a>> {
        if self.revisions.len() == 1 {
            self.revisions.iter().next().take_if(|e| e.0.is_empty()).map(|e| e.1)
        } else {
            None
        }
    }
}

#[derive(Clone, Copy)]
#[allow(dead_code)]
struct TestMetadata<'a> {
    job: &'a str,
    stage: u32,
    target: &'a str,
}

// We have to use a template for the TestGroup instead of a macro, because
// macros cannot be recursive in askama at the moment.
#[derive(Template)]
#[template(path = "test_group.askama")]
/// Represents a group of tests
struct TestGroup<'a> {
    name: String,
    /// Tests located directly in this directory
    root_tests: Vec<(String, Test<'a>)>,
    /// Nested directories with additional tests
    groups: Vec<(String, TestGroup<'a>)>,
}

impl<'a> TestGroup<'a> {
    fn test_count(&self) -> u64 {
        let root = self.root_tests.len() as u64;
        self.groups.iter().map(|(_, group)| group.test_count()).sum::<u64>() + root
    }
}

#[derive(Template)]
#[template(path = "test_suites.askama")]
struct TestSuitesPage<'a> {
    suites: TestSuites<'a>,
    test_count: u64,
}
