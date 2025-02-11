use std::collections::HashMap;
use std::path::Path;

use anyhow::Context;
use clap::Parser;
use serde_yaml::Value;

const CI_DIRECTORY: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/..");
const JOBS_YML_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../github-actions/jobs.yml");

/// Representation of a job loaded from the jobs.yml file.
#[derive(serde::Deserialize, Debug, Clone)]
struct Job {
    /// Name of the job, e.g. mingw-check
    name: String,
    /// GitHub runner on which the job should be executed
    os: String,
    env: HashMap<String, Value>,
    /// Should the job be only executed on a specific channel?
    #[serde(default)]
    only_on_channel: Option<String>,
    /// Rest of attributes that will be passed through to GitHub actions
    #[serde(flatten)]
    extra_keys: HashMap<String, Value>,
}

#[derive(serde::Deserialize, Debug)]
struct JobEnvironments {
    #[serde(rename = "pr")]
    pr_env: HashMap<String, Value>,
    #[serde(rename = "try")]
    try_env: HashMap<String, Value>,
    #[serde(rename = "auto")]
    auto_env: HashMap<String, Value>,
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
    let db = std::fs::read_to_string(path)?;
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
    name: String,
    full_name: String,
    os: String,
    env: HashMap<String, String>,
    #[serde(flatten)]
    extra_keys: HashMap<String, serde_json::Value>,
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
        if self.event_name == "pull_request" {
            return Some(RunType::PullRequest);
        } else if self.event_name == "push" {
            let is_try_build =
                ["refs/heads/try", "refs/heads/try-perf", "refs/heads/automation/bors/try"]
                    .iter()
                    .any(|r| **r == self.branch_ref);
            // Unrolled branch from a rollup for testing perf
            // This should **not** allow custom try jobs
            let is_unrolled_perf_build = self.branch_ref == "refs/heads/try-perf";
            if is_try_build {
                let custom_jobs =
                    if !is_unrolled_perf_build { Some(self.get_custom_jobs()) } else { None };
                return Some(RunType::TryJob { custom_jobs });
            }

            if self.branch_ref == "refs/heads/auto" {
                return Some(RunType::AutoJob);
            }
        }
        None
    }

    /// Tries to parse names of specific CI jobs that should be executed in the form of
    /// try-job: <job-name>
    /// from the commit message of the passed GitHub context.
    fn get_custom_jobs(&self) -> Vec<String> {
        if let Some(ref msg) = self.commit_message {
            msg.lines()
                .filter_map(|line| line.trim().strip_prefix("try-job: "))
                .map(|l| l.to_string())
                .collect()
        } else {
            vec![]
        }
    }
}

fn load_env_var(name: &str) -> anyhow::Result<String> {
    std::env::var(name).with_context(|| format!("Cannot find variable {name}"))
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

fn to_string_map(map: &HashMap<String, Value>) -> HashMap<String, String> {
    map.iter()
        .map(|(key, value)| {
            (
                key.clone(),
                serde_yaml::to_string(value)
                    .expect("Cannot serialize YAML value to string")
                    .trim()
                    .to_string(),
            )
        })
        .collect()
}

fn calculate_jobs(
    run_type: &RunType,
    db: &JobDatabase,
    channel: &str,
) -> anyhow::Result<Vec<GithubActionsJob>> {
    let (jobs, prefix, base_env) = match run_type {
        RunType::PullRequest => (db.pr_jobs.clone(), "PR", &db.envs.pr_env),
        RunType::TryJob { custom_jobs } => {
            let jobs = if let Some(custom_jobs) = custom_jobs {
                if custom_jobs.len() > 10 {
                    return Err(anyhow::anyhow!(
                        "It is only possible to schedule up to 10 custom jobs, received {} custom jobs",
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
            let mut env: HashMap<String, String> = to_string_map(base_env);
            env.extend(to_string_map(&job.env));
            let full_name = format!("{prefix} - {}", job.name);

            GithubActionsJob {
                name: job.name,
                full_name,
                os: job.os,
                env,
                extra_keys: job
                    .extra_keys
                    .into_iter()
                    .map(|(key, value)| {
                        (
                            key,
                            serde_json::to_value(&value)
                                .expect("Cannot convert extra key value from YAML to JSON"),
                        )
                    })
                    .collect(),
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

    eprintln!("Output:\njobs={jobs:?}\nrun_type={run_type}");
    println!("jobs={}", serde_json::to_string(&jobs)?);
    println!("run_type={run_type}");

    Ok(())
}

#[derive(clap::Parser)]
enum Args {
    /// Calculate a list of jobs that should be executed on CI.
    /// Should only be used on CI inside GitHub actions.
    CalculateJobMatrix,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let db = load_job_db(Path::new(JOBS_YML_PATH)).context("Cannot load jobs.yml")?;

    match args {
        Args::CalculateJobMatrix => {
            let gh_ctx = load_github_ctx()
                .context("Cannot load environment variables from GitHub Actions")?;
            let channel = std::fs::read_to_string(Path::new(CI_DIRECTORY).join("channel"))
                .context("Cannot read channel file")?;

            calculate_job_matrix(db, gh_ctx, &channel).context("Failed to calculate job matrix")?;
        }
    }

    Ok(())
}
