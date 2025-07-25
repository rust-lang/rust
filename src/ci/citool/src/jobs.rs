#[cfg(test)]
mod tests;

use std::collections::{BTreeMap, HashSet};

use anyhow::{Context as _, anyhow};
use serde_yaml::Value;

use crate::GitHubContext;
use crate::utils::load_env_var;

/// Representation of a job loaded from the `src/ci/github-actions/jobs.yml` file.
#[derive(serde::Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct Job {
    /// Name of the job, e.g. pr-check-1
    pub name: String,
    /// GitHub runner on which the job should be executed
    pub os: String,
    pub env: BTreeMap<String, Value>,
    /// Should the job be only executed on a specific channel?
    #[serde(default)]
    pub only_on_channel: Option<String>,
    /// Do not cancel the whole workflow if this job fails.
    #[serde(default)]
    pub continue_on_error: Option<bool>,
    /// Free additional disk space in the job, by removing unused packages.
    #[serde(default)]
    pub free_disk: Option<bool>,
    /// Documentation link to a resource that could help people debug this CI job.
    pub doc_url: Option<String>,
    /// Whether the job is executed on AWS CodeBuild.
    pub codebuild: Option<bool>,
}

impl Job {
    /// By default, the Docker image of a job is based on its name.
    /// However, it can be overridden by its IMAGE environment variable.
    pub fn image(&self) -> String {
        self.env
            .get("IMAGE")
            .map(|v| v.as_str().expect("IMAGE value should be a string").to_string())
            .unwrap_or_else(|| self.name.clone())
    }

    fn is_linux(&self) -> bool {
        self.os.contains("ubuntu")
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
pub struct JobDatabase {
    #[serde(rename = "pr")]
    pub pr_jobs: Vec<Job>,
    #[serde(rename = "try")]
    pub try_jobs: Vec<Job>,
    #[serde(rename = "auto")]
    pub auto_jobs: Vec<Job>,
    #[serde(rename = "optional")]
    pub optional_jobs: Vec<Job>,

    /// Shared environments for the individual run types.
    envs: JobEnvironments,
}

impl JobDatabase {
    /// Find `auto` jobs that correspond to the passed `pattern`.
    /// Patterns are matched using the glob syntax.
    /// For example `dist-*` matches all jobs starting with `dist-`.
    fn find_auto_or_optional_jobs_by_pattern(&self, pattern: &str) -> Vec<Job> {
        self.auto_jobs
            .iter()
            .chain(self.optional_jobs.iter())
            .filter(|j| glob_match::glob_match(pattern, &j.name))
            .cloned()
            .collect()
    }

    fn find_auto_job_by_name(&self, job_name: &str) -> Option<&Job> {
        self.auto_jobs.iter().find(|job| job.name == job_name)
    }
}

pub fn load_job_db(db: &str) -> anyhow::Result<JobDatabase> {
    let mut db: Value = serde_yaml::from_str(db).context("failed to parse YAML content")?;

    // We need to expand merge keys (<<), because serde_yaml can't deal with them
    // `apply_merge` only applies the merge once, so do it a few times to unwrap nested merges.

    let apply_merge = |db: &mut Value| -> anyhow::Result<()> {
        db.apply_merge().context("failed to apply merge keys")
    };

    // Apply merge twice to handle nested merges up to depth 2.
    apply_merge(&mut db)?;
    apply_merge(&mut db)?;

    let mut db: JobDatabase = serde_yaml::from_value(db).context("failed to parse job database")?;

    register_pr_jobs_as_auto_jobs(&mut db)?;

    validate_job_database(&db)?;

    Ok(db)
}

/// Maintain invariant that PR CI jobs must be a subset of Auto CI jobs modulo carve-outs.
///
/// When PR jobs are auto-registered as Auto jobs, they will have `continue_on_error` overridden to
/// be `false` to avoid wasting Auto CI resources.
///
/// When a job is already both a PR job and a auto job, we will post-validate their "equivalence
/// modulo certain carve-outs" in [`validate_job_database`].
///
/// This invariant is important to make sure that it's not easily possible (without modifying
/// `citool`) to have PRs with red PR-only CI jobs merged into `master`, causing all subsequent PR
/// CI runs to be red until the cause is fixed.
fn register_pr_jobs_as_auto_jobs(db: &mut JobDatabase) -> anyhow::Result<()> {
    for pr_job in &db.pr_jobs {
        // It's acceptable to "override" a PR job in Auto job, for instance, `x86_64-gnu-tools` will
        // receive an additional `DEPLOY_TOOLSTATES_JSON: toolstates-linux.json` env when under Auto
        // environment versus PR environment.
        if db.find_auto_job_by_name(&pr_job.name).is_some() {
            continue;
        }

        let auto_registered_job = Job { continue_on_error: Some(false), ..pr_job.clone() };
        db.auto_jobs.push(auto_registered_job);
    }

    Ok(())
}

fn validate_job_database(db: &JobDatabase) -> anyhow::Result<()> {
    fn ensure_no_duplicate_job_names(section: &str, jobs: &Vec<Job>) -> anyhow::Result<()> {
        let mut job_names = HashSet::new();
        for job in jobs {
            let job_name = job.name.as_str();
            if !job_names.insert(job_name) {
                return Err(anyhow::anyhow!(
                    "duplicate job name `{job_name}` in section `{section}`"
                ));
            }
        }
        Ok(())
    }

    ensure_no_duplicate_job_names("pr", &db.pr_jobs)?;
    ensure_no_duplicate_job_names("auto", &db.auto_jobs)?;
    ensure_no_duplicate_job_names("try", &db.try_jobs)?;
    ensure_no_duplicate_job_names("optional", &db.optional_jobs)?;

    fn equivalent_modulo_carve_out(pr_job: &Job, auto_job: &Job) -> anyhow::Result<()> {
        let Job {
            name,
            os,
            only_on_channel,
            free_disk,
            doc_url,
            codebuild,

            // Carve-out configs allowed to be different.
            env: _,
            continue_on_error: _,
        } = pr_job;

        if *name == auto_job.name
            && *os == auto_job.os
            && *only_on_channel == auto_job.only_on_channel
            && *free_disk == auto_job.free_disk
            && *doc_url == auto_job.doc_url
            && *codebuild == auto_job.codebuild
        {
            Ok(())
        } else {
            Err(anyhow!(
                "PR job `{}` differs from corresponding Auto job `{}` in configuration other than `continue_on_error` and `env`",
                pr_job.name,
                auto_job.name
            ))
        }
    }

    for pr_job in &db.pr_jobs {
        // At this point, any PR job must also be an Auto job, auto-registered or overridden.
        let auto_job = db
            .find_auto_job_by_name(&pr_job.name)
            .expect("PR job must either be auto-registered as Auto job or overridden");

        equivalent_modulo_carve_out(pr_job, auto_job)?;
    }

    // Auto CI jobs must all "fail-fast" to avoid wasting Auto CI resources. For instance, `tidy`.
    for auto_job in &db.auto_jobs {
        if auto_job.continue_on_error == Some(true) {
            return Err(anyhow!(
                "Auto job `{}` cannot have `continue_on_error: true`",
                auto_job.name
            ));
        }
    }

    Ok(())
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
    #[serde(skip_serializing_if = "Option::is_none")]
    continue_on_error: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    free_disk: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    doc_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    codebuild: Option<bool>,
}

/// Replace GitHub context variables with environment variables in job configs.
/// Used for codebuild jobs like
/// `codebuild-ubuntu-22-8c-$github.run_id-$github.run_attempt`
fn substitute_github_vars(jobs: Vec<Job>) -> anyhow::Result<Vec<Job>> {
    let run_id = load_env_var("GITHUB_RUN_ID")?;
    let run_attempt = load_env_var("GITHUB_RUN_ATTEMPT")?;

    let jobs = jobs
        .into_iter()
        .map(|mut job| {
            job.os = job
                .os
                .replace("$github.run_id", &run_id)
                .replace("$github.run_attempt", &run_attempt);
            job
        })
        .collect();

    Ok(jobs)
}

/// Skip CI jobs that are not supposed to be executed on the given `channel`.
fn skip_jobs(jobs: Vec<Job>, channel: &str) -> Vec<Job> {
    jobs.into_iter()
        .filter(|job| {
            job.only_on_channel.is_none() || job.only_on_channel.as_deref() == Some(channel)
        })
        .collect()
}

/// Type of workflow that is being executed on CI
#[derive(Debug)]
pub enum RunType {
    /// Workflows that run after a push to a PR branch
    PullRequest,
    /// Try run started with @bors try
    TryJob { job_patterns: Option<Vec<String>> },
    /// Merge attempt workflow
    AutoJob,
    /// Fake job only used for sharing Github Actions cache.
    MasterJob,
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
        RunType::TryJob { job_patterns } => {
            let jobs = if let Some(patterns) = job_patterns {
                let mut jobs: Vec<Job> = vec![];
                let mut unknown_patterns = vec![];
                for pattern in patterns {
                    let matched_jobs = db.find_auto_or_optional_jobs_by_pattern(pattern);
                    if matched_jobs.is_empty() {
                        unknown_patterns.push(pattern.clone());
                    } else {
                        for job in matched_jobs {
                            if !jobs.iter().any(|j| j.name == job.name) {
                                jobs.push(job);
                            }
                        }
                    }
                }
                if !unknown_patterns.is_empty() {
                    return Err(anyhow::anyhow!(
                        "Patterns `{}` did not match any auto jobs",
                        unknown_patterns.join(", ")
                    ));
                }
                if jobs.len() > MAX_TRY_JOBS_COUNT {
                    return Err(anyhow::anyhow!(
                        "It is only possible to schedule up to {MAX_TRY_JOBS_COUNT} custom jobs, received {} custom jobs expanded from {} pattern(s)",
                        jobs.len(),
                        patterns.len()
                    ));
                }
                jobs
            } else {
                db.try_jobs.clone()
            };
            (jobs, "try", &db.envs.try_env)
        }
        RunType::AutoJob => (db.auto_jobs.clone(), "auto", &db.envs.auto_env),
        RunType::MasterJob => return Ok(vec![]),
    };
    let jobs = substitute_github_vars(jobs.clone())
        .context("Failed to substitute GitHub context variables in jobs")?;
    let jobs = skip_jobs(jobs, channel);
    let jobs = jobs
        .into_iter()
        .map(|job| {
            let mut env: BTreeMap<String, serde_json::Value> = crate::yaml_map_to_json(base_env);
            env.extend(crate::yaml_map_to_json(&job.env));
            let full_name = format!("{prefix} - {}", job.name);

            // When the default `@bors try` job is executed (which is usually done
            // for benchmarking performance, running crater or for downloading the
            // built toolchain using `rustup-toolchain-install-master`),
            // we inject the `DIST_TRY_BUILD` environment variable to the jobs
            // to tell `opt-dist` to make the build faster by skipping certain steps.
            if let RunType::TryJob { job_patterns } = run_type {
                if job_patterns.is_none() {
                    env.insert(
                        "DIST_TRY_BUILD".to_string(),
                        serde_json::value::Value::Number(1.into()),
                    );
                }
            }

            GithubActionsJob {
                name: job.name,
                full_name,
                os: job.os,
                env,
                continue_on_error: job.continue_on_error,
                free_disk: job.free_disk,
                doc_url: job.doc_url,
                codebuild: job.codebuild,
            }
        })
        .collect();

    Ok(jobs)
}

pub fn calculate_job_matrix(
    db: JobDatabase,
    gh_ctx: GitHubContext,
    channel: &str,
) -> anyhow::Result<()> {
    let run_type = gh_ctx.get_run_type().ok_or_else(|| {
        anyhow::anyhow!("Cannot determine the type of workflow that is being executed")
    })?;
    eprintln!("Run type: {run_type:?}");

    let jobs = calculate_jobs(&run_type, &db, channel)?;
    if jobs.is_empty() && !matches!(run_type, RunType::MasterJob) {
        return Err(anyhow::anyhow!("Computed job list is empty"));
    }

    let run_type = match run_type {
        RunType::PullRequest => "pr",
        RunType::TryJob { .. } => "try",
        RunType::AutoJob => "auto",
        RunType::MasterJob => "master",
    };

    eprintln!("Output");
    eprintln!("jobs={jobs:?}");
    eprintln!("run_type={run_type}");
    println!("jobs={}", serde_json::to_string(&jobs)?);
    println!("run_type={run_type}");

    Ok(())
}

pub fn find_linux_job<'a>(jobs: &'a [Job], name: &str) -> anyhow::Result<&'a Job> {
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
