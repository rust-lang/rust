use std::fs;

use anyhow::{Context, Result};
use camino::Utf8PathBuf;

use crate::model::AnalysisFindings;

pub fn write_findings_json(analysis_dir: &Utf8PathBuf, findings: &AnalysisFindings) -> Result<()> {
    let body = serde_json::to_vec_pretty(findings).context("failed to serialize findings json")?;
    fs::write(analysis_dir.join("findings.json"), body)
        .with_context(|| format!("failed to write {analysis_dir}/findings.json"))?;
    Ok(())
}

pub fn load_findings_json(analysis_dir: &Utf8PathBuf) -> Result<AnalysisFindings> {
    let data = fs::read(analysis_dir.join("findings.json"))
        .with_context(|| format!("failed to read {analysis_dir}/findings.json"))?;
    serde_json::from_slice(&data).context("failed to parse findings.json")
}
