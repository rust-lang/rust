use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

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

    struct WorkItem {
        job: String,
    }

    struct WorkResult {
        job: String,
        data: anyhow::Result<JsonRoot>,
        parent_data: Option<anyhow::Result<JsonRoot>>,
    }

    // Avoid overloading the GitHub API
    let thread_count = std::thread::available_parallelism().map(|v| v.into()).unwrap_or(4).min(4);

    // Ensure that we have enough capacity to submit all job download requests, to avoid having to
    // interleave submission of work and reading of results.
    let (submit_tx, submit_rx) = std::sync::mpsc::sync_channel::<WorkItem>(job_db.auto_jobs.len());
    let submit_rx = Arc::new(Mutex::new(submit_rx));

    let (result_tx, result_rx) = std::sync::mpsc::sync_channel::<WorkResult>(thread_count);

    let start = Instant::now();
    // There are many jobs to download, so we do it in parallel
    // To avoid adding more dependencies, we create our own little thread pool.
    std::thread::scope(|s| {
        for _ in 0..thread_count {
            let submit_rx = submit_rx.clone();
            let result_tx = result_tx.clone();
            s.spawn(move || {
                loop {
                    let item = {
                        let Ok(item) = submit_rx.lock().unwrap().recv() else {
                            break;
                        };
                        item
                    };
                    eprintln!("Downloading metrics of job {}", item.job);
                    let data = download_job_metrics(&item.job, current);
                    let parent_data = parent.map(|sha| download_job_metrics(&item.job, &sha));
                    let result = WorkResult { job: item.job, data, parent_data };

                    // If the receiver was dropped, end processing downloads
                    if result_tx.send(result).is_err() {
                        break;
                    }
                }
            });
        }

        // Submit all jobs
        for job in &job_db.auto_jobs {
            let item = WorkItem { job: job.name.to_string() };
            submit_tx.send(item).unwrap();
        }
        drop(result_tx);
        drop(submit_tx);

        // Wait for results
        for result in result_rx {
            let parent = match result.parent_data {
                Some(Ok(data)) => Some(data),
                Some(Err(error)) => {
                    eprintln!(
                        r#"Did not find parent metrics for job `{}`: {error:?}.
Maybe it was newly added?"#,
                        result.job
                    );
                    None
                }
                None => None,
            };

            jobs.insert(
                result.job.clone(),
                JobMetrics {
                    parent,
                    current: result.data.with_context(|| {
                        anyhow::anyhow!("Could not download metrics for job `{}`", result.job)
                    })?,
                },
            );
        }

        anyhow::Ok(())
    })?;
    eprintln!("Download finished in {:.2}", start.elapsed().as_secs_f64());

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
