mod cpu_usage;
mod datadog;
mod jobs;
mod merge_report;
mod metrics;
mod utils;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::Context;
use clap::Parser;
use jobs::JobDatabase;
use serde_yaml::Value;

use crate::cpu_usage::load_cpu_usage;
use crate::datadog::upload_datadog_metric;
use crate::jobs::RunType;
use crate::merge_report::post_merge_report;
use crate::metrics::postprocess_metrics;
use crate::utils::load_env_var;

const CI_DIRECTORY: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/..");
const DOCKER_DIRECTORY: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../docker");
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
    /// Postprocess the metrics.json file generated by bootstrap.
    PostprocessMetrics {
        /// Path to the metrics.json file
        metrics_path: PathBuf,
        /// Path to a file where the postprocessed metrics summary will be stored.
        /// Usually, this will be GITHUB_STEP_SUMMARY on CI.
        summary_path: PathBuf,
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
        Args::PostprocessMetrics { metrics_path, summary_path } => {
            postprocess_metrics(&metrics_path, &summary_path)?;
        }
        Args::PostMergeReport { current: commit, parent } => {
            post_merge_report(load_db(default_jobs_file)?, parent, commit)?;
        }
    }

    Ok(())
}
