use std::collections::HashMap;

use anyhow::Context;
use build_helper::metrics::{CiMetadata, JsonRoot};

pub struct GitHubClient;

impl GitHubClient {
    fn get_workflow_run_jobs(
        &self,
        repo: &str,
        workflow_run_id: u64,
    ) -> anyhow::Result<Vec<GitHubJob>> {
        let req = ureq::get(format!(
            "https://api.github.com/repos/{repo}/actions/runs/{workflow_run_id}/jobs?per_page=100"
        ))
        .header("User-Agent", "rust-lang/rust/citool")
        .header("Accept", "application/vnd.github+json")
        .header("X-GitHub-Api-Version", "2022-11-28")
        .call()
        .context("cannot get workflow job list")?;

        let status = req.status();
        let mut body = req.into_body();
        if status.is_success() {
            // This API response is actually paged, but we assume for now that there are at
            // most 100 jobs per workflow.
            let response = body
                .read_json::<WorkflowRunJobsResponse>()
                .context("cannot deserialize workflow run jobs response")?;
            // The CI job names have a prefix, e.g. `auto - foo`. We remove the prefix here to
            // normalize the job name.
            Ok(response
                .jobs
                .into_iter()
                .map(|mut job| {
                    job.name = job
                        .name
                        .split_once(" - ")
                        .map(|res| res.1.to_string())
                        .unwrap_or_else(|| job.name);
                    job
                })
                .collect())
        } else {
            Err(anyhow::anyhow!(
                "Cannot get jobs of workflow run {workflow_run_id}: {status}\n{}",
                body.read_to_string()?
            ))
        }
    }
}

#[derive(serde::Deserialize)]
struct WorkflowRunJobsResponse {
    jobs: Vec<GitHubJob>,
}

#[derive(serde::Deserialize)]
struct GitHubJob {
    name: String,
    id: u64,
}

/// Can be used to resolve information about GitHub Actions jobs.
/// Caches results internally to avoid too unnecessary GitHub API calls.
pub struct JobInfoResolver {
    client: GitHubClient,
    // Workflow run ID -> jobs
    workflow_job_cache: HashMap<u64, Vec<GitHubJob>>,
}

impl JobInfoResolver {
    pub fn new() -> Self {
        Self { client: GitHubClient, workflow_job_cache: Default::default() }
    }

    /// Get a link to a job summary for the given job name and bootstrap execution.
    pub fn get_job_summary_link(&mut self, job_name: &str, metrics: &JsonRoot) -> Option<String> {
        metrics.ci_metadata.as_ref().and_then(|metadata| {
            self.get_job_id(metadata, job_name).map(|job_id| {
                format!(
                    "https://github.com/{}/actions/runs/{}#summary-{job_id}",
                    metadata.repository, metadata.workflow_run_id
                )
            })
        })
    }

    fn get_job_id(&mut self, ci_metadata: &CiMetadata, job_name: &str) -> Option<u64> {
        if let Some(job) = self
            .workflow_job_cache
            .get(&ci_metadata.workflow_run_id)
            .and_then(|jobs| jobs.iter().find(|j| j.name == job_name))
        {
            return Some(job.id);
        }

        let jobs = self
            .client
            .get_workflow_run_jobs(&ci_metadata.repository, ci_metadata.workflow_run_id)
            .inspect_err(|e| eprintln!("Cannot download workflow jobs: {e:?}"))
            .ok()?;
        let job_id = jobs.iter().find(|j| j.name == job_name).map(|j| j.id);
        // Save the cache even if the job name was not found, it could be useful for further lookups
        self.workflow_job_cache.insert(ci_metadata.workflow_run_id, jobs);
        job_id
    }
}
