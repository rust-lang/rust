mod analysis;
mod cpu_usage;
mod datadog;
mod github;
mod jobs;
mod metrics;
mod test_dashboard;
mod utils;

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::process::Command;

use analysis::output_bootstrap_stats;
use anyhow::Context;
use clap::Parser;
use jobs::JobDatabase;
use serde_yaml::Value;

use crate::analysis::{output_largest_duration_changes, output_test_diffs};
use crate::cpu_usage::load_cpu_usage;
use crate::datadog::upload_datadog_metric;
use crate::github::JobInfoResolver;
use crate::jobs::RunType;
use crate::metrics::{JobMetrics, download_auto_job_metrics, download_job_metrics, load_metrics};
use crate::test_dashboard::generate_test_dashboard;
use crate::utils::{load_env_var, output_details};

const CI_DIRECTORY: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/..");
pub const DOCKER_DIRECTORY: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../docker");
const JOBS_YML_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../github-actions/jobs.yml");

struct GitHubContext {
    event_name: String,
    branch_ref: String,
    commit_message: Option<String>,
}

impl GitHubContext {
    fn get_run_type(&self) -> Option<RunType> {
        match (self.event_name.as_str(), self.branch_ref.as_str()) {
            ("pull_request", _) => Some(RunType::PullRequest),
            ("push", "refs/heads/try-perf") => Some(RunType::TryJob { job_patterns: None }),
            ("push", "refs/heads/try" | "refs/heads/automation/bors/try") => {
                let patterns = self.get_try_job_patterns();
                let patterns = if !patterns.is_empty() { Some(patterns) } else { None };
                Some(RunType::TryJob { job_patterns: patterns })
            }
            ("push", "refs/heads/auto") => Some(RunType::AutoJob),
            _ => None,
        }
    }

    /// Tries to parse patterns of CI jobs that should be executed
    /// from the commit message of the passed GitHub context
    ///
    /// They can be specified in the form of
    /// try-job: <job-pattern>
    /// or
    /// try-job: `<job-pattern>`
    /// (to avoid GitHub rendering the glob patterns as Markdown)
    fn get_try_job_patterns(&self) -> Vec<String> {
        if let Some(ref msg) = self.commit_message {
            msg.lines()
                .filter_map(|line| line.trim().strip_prefix("try-job: "))
                // Strip backticks if present
                .map(|l| l.trim_matches('`'))
                .map(|l| l.trim().to_string())
                .collect()
        } else {
            vec![]
        }
    }
}

fn load_github_ctx() -> anyhow::Result<GitHubContext> {
    let event_name = load_env_var("GITHUB_EVENT_NAME")?;
    let commit_message =
        if event_name == "push" { Some(load_env_var("COMMIT_MESSAGE")?) } else { None };

    Ok(GitHubContext { event_name, branch_ref: load_env_var("GITHUB_REF")?, commit_message })
}

fn yaml_map_to_json(map: &BTreeMap<String, Value>) -> BTreeMap<String, serde_json::Value> {
    map.into_iter()
        .map(|(key, value)| {
            (
                key.clone(),
                serde_json::to_value(&value).expect("Cannot convert map value from YAML to JSON"),
            )
        })
        .collect()
}

fn run_workflow_locally(db: JobDatabase, job_type: JobType, name: String) -> anyhow::Result<()> {
    let jobs = match job_type {
        JobType::Auto => &db.auto_jobs,
        JobType::PR => &db.pr_jobs,
    };
    let job =
        jobs::find_linux_job(jobs, &name).with_context(|| format!("Cannot find job {name}"))?;

    let mut custom_env: BTreeMap<String, String> = BTreeMap::new();
    // Replicate src/ci/scripts/setup-environment.sh
    // Adds custom environment variables to the job
    if name.starts_with("dist-") {
        if name.ends_with("-alt") {
            custom_env.insert("DEPLOY_ALT".to_string(), "1".to_string());
        } else {
            custom_env.insert("DEPLOY".to_string(), "1".to_string());
        }
    }
    custom_env.extend(job.env.iter().map(|(key, value)| {
        let value = match value {
            Value::Bool(value) => value.to_string(),
            Value::Number(value) => value.to_string(),
            Value::String(value) => value.clone(),
            _ => panic!("Unexpected type for environment variable {key} Only bool/number/string is supported.")
        };
        (key.clone(), value)
    }));

    let mut cmd = Command::new(Path::new(DOCKER_DIRECTORY).join("run.sh"));
    cmd.arg(job.image());
    cmd.envs(custom_env);

    eprintln!("Executing {cmd:?}");

    let result = cmd.spawn()?.wait()?;
    if !result.success() { Err(anyhow::anyhow!("Job failed")) } else { Ok(()) }
}

fn upload_ci_metrics(cpu_usage_csv: &Path) -> anyhow::Result<()> {
    let usage = load_cpu_usage(cpu_usage_csv).context("Cannot load CPU usage from input CSV")?;
    eprintln!("CPU usage\n{usage:?}");

    let avg = if !usage.is_empty() { usage.iter().sum::<f64>() / usage.len() as f64 } else { 0.0 };
    eprintln!("CPU usage average: {avg}");

    upload_datadog_metric("avg-cpu-usage", avg).context("Cannot upload Datadog metric")?;

    Ok(())
}

fn postprocess_metrics(
    metrics_path: PathBuf,
    parent: Option<String>,
    job_name: Option<String>,
) -> anyhow::Result<()> {
    let metrics = load_metrics(&metrics_path)?;

    let mut job_info_resolver = JobInfoResolver::new();
    if let (Some(parent), Some(job_name)) = (parent, job_name) {
        // This command is executed also on PR builds, which might not have parent metrics
        // available, because some PR jobs don't run on auto builds, and PR jobs do not upload metrics
        // due to missing permissions.
        // To avoid having to detect if this is a PR job, and to avoid having failed steps in PR jobs,
        // we simply print an error if the parent metrics were not found, but otherwise exit
        // successfully.
        match download_job_metrics(&job_name, &parent).context("cannot download parent metrics") {
            Ok(parent_metrics) => {
                output_bootstrap_stats(&metrics, Some(&parent_metrics));

                let job_metrics = HashMap::from([(
                    job_name,
                    JobMetrics { parent: Some(parent_metrics), current: metrics },
                )]);
                output_test_diffs(&job_metrics, &mut job_info_resolver);
                return Ok(());
            }
            Err(error) => {
                eprintln!(
                    "Metrics for job `{job_name}` and commit `{parent}` not found: {error:?}"
                );
            }
        }
    }

    output_bootstrap_stats(&metrics, None);

    Ok(())
}

fn post_merge_report(db: JobDatabase, current: String, parent: String) -> anyhow::Result<()> {
    let metrics = download_auto_job_metrics(&db, Some(&parent), &current)?;

    println!("\nComparing {parent} (parent) -> {current} (this PR)\n");

    let mut job_info_resolver = JobInfoResolver::new();
    output_test_diffs(&metrics, &mut job_info_resolver);

    output_details("Test dashboard", || {
        println!(
            r#"Run

```bash
cargo run --manifest-path src/ci/citool/Cargo.toml -- \
    test-dashboard {current} --output-dir test-dashboard
```
And then open `test-dashboard/index.html` in your browser to see an overview of all executed tests.
"#
        );
    });

    output_largest_duration_changes(&metrics, &mut job_info_resolver);

    Ok(())
}

#[derive(clap::Parser)]
enum Args {
    /// Calculate a list of jobs that should be executed on CI.
    /// Should only be used on CI inside GitHub actions.
    CalculateJobMatrix {
        #[clap(long)]
        jobs_file: Option<PathBuf>,
    },
    /// Execute a given CI job locally.
    #[clap(name = "run-local")]
    RunJobLocally {
        /// Name of the job that should be executed.
        name: String,
        /// Type of the job that should be executed.
        #[clap(long = "type", default_value = "auto")]
        job_type: JobType,
    },
    /// Postprocess the metrics.json file generated by bootstrap and output
    /// various statistics.
    /// If `--parent` and `--job-name` are provided, also display a diff
    /// against previous metrics that are downloaded from CI.
    PostprocessMetrics {
        /// Path to the metrics.json file
        metrics_path: PathBuf,
        /// A parent SHA against which to compare.
        #[clap(long, requires("job_name"))]
        parent: Option<String>,
        /// The name of the current job.
        #[clap(long, requires("parent"))]
        job_name: Option<String>,
    },
    /// Upload CI metrics to Datadog.
    UploadBuildMetrics {
        /// Path to a CSV containing the CI job CPU usage.
        cpu_usage_csv: PathBuf,
    },
    /// Generate a report of test execution changes between two rustc commits.
    PostMergeReport {
        /// Parent commit to use as a base of the comparison.
        parent: String,
        /// Current commit that will be compared to `parent`.
        current: String,
    },
    /// Generate a directory containing a HTML dashboard of test results from a CI run.
    TestDashboard {
        /// Commit SHA that was tested on CI to analyze.
        current: String,
        /// Output path for the HTML directory.
        #[clap(long)]
        output_dir: PathBuf,
    },
}

#[derive(clap::ValueEnum, Clone)]
pub enum JobType {
    /// Merge attempt ("auto") job
    Auto,
    /// Pull request job
    PR,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let default_jobs_file = Path::new(JOBS_YML_PATH);
    let load_db = |jobs_path| {
        let db = utils::read_to_string(jobs_path)?;
        Ok::<_, anyhow::Error>(jobs::load_job_db(&db).context("Cannot load jobs.yml")?)
    };

    match args {
        Args::CalculateJobMatrix { jobs_file } => {
            let jobs_path = jobs_file.as_deref().unwrap_or(default_jobs_file);
            let gh_ctx = load_github_ctx()
                .context("Cannot load environment variables from GitHub Actions")?;
            let channel = utils::read_to_string(Path::new(CI_DIRECTORY).join("channel"))
                .context("Cannot read channel file")?
                .trim()
                .to_string();

            jobs::calculate_job_matrix(load_db(jobs_path)?, gh_ctx, &channel)
                .context("Failed to calculate job matrix")?;
        }
        Args::RunJobLocally { job_type, name } => {
            run_workflow_locally(load_db(default_jobs_file)?, job_type, name)?;
        }
        Args::UploadBuildMetrics { cpu_usage_csv } => {
            upload_ci_metrics(&cpu_usage_csv)?;
        }
        Args::PostprocessMetrics { metrics_path, parent, job_name } => {
            postprocess_metrics(metrics_path, parent, job_name)?;
        }
        Args::PostMergeReport { current, parent } => {
            post_merge_report(load_db(&default_jobs_file)?, current, parent)?;
        }
        Args::TestDashboard { current, output_dir } => {
            let db = load_db(&default_jobs_file)?;
            generate_test_dashboard(db, &current, &output_dir)?;
        }
    }

    Ok(())
}
