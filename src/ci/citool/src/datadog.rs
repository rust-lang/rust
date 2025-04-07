use anyhow::Context;

use crate::utils::load_env_var;

/// Uploads a custom CI pipeline metric to Datadog.
/// Expects to be executed from within the context of a GitHub Actions job.
pub fn upload_datadog_metric(name: &str, value: f64) -> anyhow::Result<()> {
    let datadog_api_key = load_env_var("DATADOG_API_KEY")?;
    let github_server_url = load_env_var("GITHUB_SERVER_URL")?;
    let github_repository = load_env_var("GITHUB_REPOSITORY")?;
    let github_run_id = load_env_var("GITHUB_RUN_ID")?;
    let github_run_attempt = load_env_var("GITHUB_RUN_ATTEMPT")?;
    let github_job = load_env_var("GITHUB_JOB")?;
    let dd_github_job_name = load_env_var("DD_GITHUB_JOB_NAME")?;

    // This API endpoint is not documented in Datadog's API reference currently.
    // It was reverse-engineered from the `datadog-ci measure` npm command.
    ureq::post("https://api.datadoghq.com/api/v2/ci/pipeline/metrics")
        .header("DD-API-KEY", datadog_api_key)
        .send_json(serde_json::json!({
            "data": {
                "attributes": {
                    "ci_env": {
                        "GITHUB_SERVER_URL": github_server_url,
                        "GITHUB_REPOSITORY": github_repository,
                        "GITHUB_RUN_ID": github_run_id,
                        "GITHUB_RUN_ATTEMPT": github_run_attempt,
                        "GITHUB_JOB": github_job,
                        "DD_GITHUB_JOB_NAME": dd_github_job_name
                    },
                    // Job level
                    "ci_level": 1,
                    "metrics": {
                        name: value
                    },
                    "provider": "github"
                },
                "type": "ci_custom_metric"
            }
        }))
        .context("cannot send metric to DataDog")?;
    Ok(())
}
