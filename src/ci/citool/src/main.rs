mod cpu_usage;
mod datadog;
mod merge_report;
mod metrics;
mod utils;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::Context;
use clap::Parser;
use serde_yaml::Value;

use crate::cpu_usage::load_cpu_usage;
use crate::datadog::upload_datadog_metric;
use crate::merge_report::post_merge_report;
use crate::metrics::postprocess_metrics;
use crate::utils::load_env_var;

const CI_DIRECTORY: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/..");
const DOCKER_DIRECTORY: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../docker");
const JOBS_YML_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../github-actions/jobs.yml");

/// Representation of a job loaded from the `src/ci/github-actions/jobs.yml` file.
#[derive(serde::Deserialize, Debug, Clone)]
struct Job {
    /// Name of the job, e.g. mingw-check
    name: String,
    /// GitHub runner on which the job should be executed
    os: String,
    env: BTreeMap<String, Value>,
    /// Should the job be only executed on a specific channel?
    #[serde(default)]
    only_on_channel: Option<String>,
    /// Rest of attributes that will be passed through to GitHub actions
    #[serde(flatten)]
    extra_keys: BTreeMap<String, Value>,
}

impl Job {
    fn is_linux(&self) -> bool {
        self.os.contains("ubuntu")
    }

    /// By default, the Docker image of a job is based on its name.
    /// However, it can be overridden by its IMAGE environment variable.
    fn image(&self) -> String {
        self.env
            .get("IMAGE")
            .map(|v| v.as_str().expect("IMAGE value should be a string").to_string())
            .unwrap_or_else(|| self.name.clone())
    }
}

#[derive(serde::Deserialize, Debug)]
struct JobEnvironments {
    #[serde(rename = "pr")]
    pr_env: BTreeMap<String, Value>,
    #[serde(rename = "try")]
    try_env: BTreeMap<String, Value>,
    #[serde(rename = "auto")]
    auto_env: BTreeMap<String, Value>,
}

#[derive(serde::Deserialize, Debug)]
struct JobDatabase {
    #[serde(rename = "pr")]
    pr_jobs: Vec<Job>,
    #[serde(rename = "try")]
    try_jobs: Vec<Job>,
    #[serde(rename = "auto")]
    auto_jobs: Vec<Job>,

    /// Shared environments for the individual run types.
    envs: JobEnvironments,
}

impl JobDatabase {
    fn find_auto_job_by_name(&self, name: &str) -> Option<Job> {
        self.auto_jobs.iter().find(|j| j.name == name).cloned()
    }
}

fn load_job_db(path: &Path) -> anyhow::Result<JobDatabase> {
    let db = utils::read_to_string(path)?;
    let mut db: Value = serde_yaml::from_str(&db)?;

    // We need to expand merge keys (<<), because serde_yaml can't deal with them
    // `apply_merge` only applies the merge once, so do it a few times to unwrap nested merges.
    db.apply_merge()?;
    db.apply_merge()?;

    let db: JobDatabase = serde_yaml::from_value(db)?;
    Ok(db)
}

/// Representation of a job outputted to a GitHub Actions workflow.
#[derive(serde::Serialize, Debug)]
struct GithubActionsJob {
    /// The main identifier of the job, used by CI scripts to determine what should be executed.
    name: String,
    /// Helper label displayed in GitHub Actions interface, containing the job name and a run type
    /// prefix (PR/try/auto).
    full_name: String,
    os: String,
    env: BTreeMap<String, serde_json::Value>,
    #[serde(flatten)]
    extra_keys: BTreeMap<String, serde_json::Value>,
}

/// Type of workflow that is being executed on CI
#[derive(Debug)]
enum RunType {
    /// Workflows that run after a push to a PR branch
    PullRequest,
    /// Try run started with @bors try
    TryJob { custom_jobs: Option<Vec<String>> },
    /// Merge attempt workflow
    AutoJob,
}

struct GitHubContext {
    event_name: String,
    branch_ref: String,
    commit_message: Option<String>,
}

impl GitHubContext {
    fn get_run_type(&self) -> Option<RunType> {
        match (self.event_name.as_str(), self.branch_ref.as_str()) {
            ("pull_request", _) => Some(RunType::PullRequest),
            ("push", "refs/heads/try-perf") => Some(RunType::TryJob { custom_jobs: None }),
            ("push", "refs/heads/try" | "refs/heads/automation/bors/try") => {
                let custom_jobs = self.get_custom_jobs();
                let custom_jobs = if !custom_jobs.is_empty() { Some(custom_jobs) } else { None };
                Some(RunType::TryJob { custom_jobs })
            }
            ("push", "refs/heads/auto") => Some(RunType::AutoJob),
            _ => None,
        }
    }

    /// Tries to parse names of specific CI jobs that should be executed in the form of
    /// try-job: <job-name>
    /// from the commit message of the passed GitHub context.
    fn get_custom_jobs(&self) -> Vec<String> {
        if let Some(ref msg) = self.commit_message {
            msg.lines()
                .filter_map(|line| line.trim().strip_prefix("try-job: "))
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

/// Skip CI jobs that are not supposed to be executed on the given `channel`.
fn skip_jobs(jobs: Vec<Job>, channel: &str) -> Vec<Job> {
    jobs.into_iter()
        .filter(|job| {
            job.only_on_channel.is_none() || job.only_on_channel.as_deref() == Some(channel)
        })
        .collect()
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

/// Maximum number of custom try jobs that can be requested in a single
/// `@bors try` request.
const MAX_TRY_JOBS_COUNT: usize = 20;

fn calculate_jobs(
    run_type: &RunType,
    db: &JobDatabase,
    channel: &str,
) -> anyhow::Result<Vec<GithubActionsJob>> {
    let (jobs, prefix, base_env) = match run_type {
        RunType::PullRequest => (db.pr_jobs.clone(), "PR", &db.envs.pr_env),
        RunType::TryJob { custom_jobs } => {
            let jobs = if let Some(custom_jobs) = custom_jobs {
                if custom_jobs.len() > MAX_TRY_JOBS_COUNT {
                    return Err(anyhow::anyhow!(
                        "It is only possible to schedule up to {MAX_TRY_JOBS_COUNT} custom jobs, received {} custom jobs",
                        custom_jobs.len()
                    ));
                }

                let mut jobs = vec![];
                let mut unknown_jobs = vec![];
                for custom_job in custom_jobs {
                    if let Some(job) = db.find_auto_job_by_name(custom_job) {
                        jobs.push(job);
                    } else {
                        unknown_jobs.push(custom_job.clone());
                    }
                }
                if !unknown_jobs.is_empty() {
                    return Err(anyhow::anyhow!(
                        "Custom job(s) `{}` not found in auto jobs",
                        unknown_jobs.join(", ")
                    ));
                }
                jobs
            } else {
                db.try_jobs.clone()
            };
            (jobs, "try", &db.envs.try_env)
        }
        RunType::AutoJob => (db.auto_jobs.clone(), "auto", &db.envs.auto_env),
    };
    let jobs = skip_jobs(jobs, channel);
    let jobs = jobs
        .into_iter()
        .map(|job| {
            let mut env: BTreeMap<String, serde_json::Value> = yaml_map_to_json(base_env);
            env.extend(yaml_map_to_json(&job.env));
            let full_name = format!("{prefix} - {}", job.name);

            GithubActionsJob {
                name: job.name,
                full_name,
                os: job.os,
                env,
                extra_keys: yaml_map_to_json(&job.extra_keys),
            }
        })
        .collect();

    Ok(jobs)
}

fn calculate_job_matrix(
    db: JobDatabase,
    gh_ctx: GitHubContext,
    channel: &str,
) -> anyhow::Result<()> {
    let run_type = gh_ctx.get_run_type().ok_or_else(|| {
        anyhow::anyhow!("Cannot determine the type of workflow that is being executed")
    })?;
    eprintln!("Run type: {run_type:?}");

    let jobs = calculate_jobs(&run_type, &db, channel)?;
    if jobs.is_empty() {
        return Err(anyhow::anyhow!("Computed job list is empty"));
    }

    let run_type = match run_type {
        RunType::PullRequest => "pr",
        RunType::TryJob { .. } => "try",
        RunType::AutoJob => "auto",
    };

    eprintln!("Output");
    eprintln!("jobs={jobs:?}");
    eprintln!("run_type={run_type}");
    println!("jobs={}", serde_json::to_string(&jobs)?);
    println!("run_type={run_type}");

    Ok(())
}

fn find_linux_job<'a>(jobs: &'a [Job], name: &str) -> anyhow::Result<&'a Job> {
    let Some(job) = jobs.iter().find(|j| j.name == name) else {
        let available_jobs: Vec<&Job> = jobs.iter().filter(|j| j.is_linux()).collect();
        let mut available_jobs =
            available_jobs.iter().map(|j| j.name.to_string()).collect::<Vec<_>>();
        available_jobs.sort();
        return Err(anyhow::anyhow!(
            "Job {name} not found. The following jobs are available:\n{}",
            available_jobs.join(", ")
        ));
    };
    if !job.is_linux() {
        return Err(anyhow::anyhow!("Only Linux jobs can be executed locally"));
    }

    Ok(job)
}

fn run_workflow_locally(db: JobDatabase, job_type: JobType, name: String) -> anyhow::Result<()> {
    let jobs = match job_type {
        JobType::Auto => &db.auto_jobs,
        JobType::PR => &db.pr_jobs,
    };
    let job = find_linux_job(jobs, &name).with_context(|| format!("Cannot find job {name}"))?;

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
enum JobType {
    /// Merge attempt ("auto") job
    Auto,
    /// Pull request job
    PR,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let default_jobs_file = Path::new(JOBS_YML_PATH);
    let load_db = |jobs_path| load_job_db(jobs_path).context("Cannot load jobs.yml");

    match args {
        Args::CalculateJobMatrix { jobs_file } => {
            let jobs_path = jobs_file.as_deref().unwrap_or(default_jobs_file);
            let gh_ctx = load_github_ctx()
                .context("Cannot load environment variables from GitHub Actions")?;
            let channel = utils::read_to_string(Path::new(CI_DIRECTORY).join("channel"))
                .context("Cannot read channel file")?
                .trim()
                .to_string();

            calculate_job_matrix(load_db(jobs_path)?, gh_ctx, &channel)
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
