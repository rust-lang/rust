use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::Context;
use build_helper::metrics::{JsonNode, JsonRoot, TestSuite};

use crate::jobs::JobDatabase;

pub type JobName = String;

pub fn get_test_suites(metrics: &JsonRoot) -> Vec<&TestSuite> {
    fn visit_test_suites<'a>(nodes: &'a [JsonNode], suites: &mut Vec<&'a TestSuite>) {
        for node in nodes {
            match node {
                JsonNode::RustbuildStep { children, .. } => {
                    visit_test_suites(&children, suites);
                }
                JsonNode::TestSuite(suite) => {
                    suites.push(&suite);
                }
            }
        }
    }

    let mut suites = vec![];
    for invocation in &metrics.invocations {
        visit_test_suites(&invocation.children, &mut suites);
    }
    suites
}

pub fn load_metrics(path: &Path) -> anyhow::Result<JsonRoot> {
    let metrics = std::fs::read_to_string(path)
        .with_context(|| format!("Cannot read JSON metrics from {path:?}"))?;
    let metrics: JsonRoot = serde_json::from_str(&metrics)
        .with_context(|| format!("Cannot deserialize JSON metrics from {path:?}"))?;
    Ok(metrics)
}

pub struct JobMetrics {
    pub parent: Option<JsonRoot>,
    pub current: JsonRoot,
}

/// Download before/after metrics for all auto jobs in the job database.
/// `parent` and `current` should be commit SHAs.
pub fn download_auto_job_metrics(
    job_db: &JobDatabase,
    parent: Option<&str>,
    current: &str,
) -> anyhow::Result<HashMap<JobName, JobMetrics>> {
    let mut jobs = HashMap::default();

    for job in &job_db.auto_jobs {
        eprintln!("Downloading metrics of job {}", job.name);
        let metrics_parent =
            parent.and_then(|parent| match download_job_metrics(&job.name, parent) {
                Ok(metrics) => Some(metrics),
                Err(error) => {
                    eprintln!(
                        r#"Did not find metrics for job `{}` at `{parent}`: {error:?}.
Maybe it was newly added?"#,
                        job.name
                    );
                    None
                }
            });
        let metrics_current = download_job_metrics(&job.name, current)?;
        jobs.insert(
            job.name.clone(),
            JobMetrics { parent: metrics_parent, current: metrics_current },
        );
    }
    Ok(jobs)
}

pub fn download_job_metrics(job_name: &str, sha: &str) -> anyhow::Result<JsonRoot> {
    // Best effort cache to speed-up local re-executions of citool
    let cache_path = PathBuf::from(".citool-cache").join(sha).join(format!("{job_name}.json"));
    if cache_path.is_file() {
        if let Ok(metrics) = std::fs::read_to_string(&cache_path)
            .map_err(|err| err.into())
            .and_then(|data| anyhow::Ok::<JsonRoot>(serde_json::from_str::<JsonRoot>(&data)?))
        {
            return Ok(metrics);
        }
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

    if let Ok(_) = std::fs::create_dir_all(cache_path.parent().unwrap()) {
        if let Ok(data) = serde_json::to_string(&data) {
            let _ = std::fs::write(cache_path, data);
        }
    }

    Ok(data)
}

fn get_metrics_url(job_name: &str, sha: &str) -> String {
    let suffix = if job_name.ends_with("-alt") { "-alt" } else { "" };
    format!("https://ci-artifacts.rust-lang.org/rustc-builds{suffix}/{sha}/metrics-{job_name}.json")
}
