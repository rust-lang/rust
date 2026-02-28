pub mod html;
pub mod json;

use anyhow::Result;
use camino::Utf8PathBuf;

use crate::diff::load_manifest;
use crate::model::AnalysisFindings;

pub fn write_reports(
    analysis_dir: &Utf8PathBuf,
    manifest: &crate::model::DiffManifest,
    findings: &AnalysisFindings,
) -> Result<()> {
    json::write_findings_json(analysis_dir, findings)?;
    html::write_html_report(analysis_dir, manifest, findings)?;
    Ok(())
}

pub fn regenerate_reports(analysis_dir: &Utf8PathBuf) -> Result<()> {
    let manifest = load_manifest(analysis_dir)?;
    let findings = json::load_findings_json(analysis_dir)?;
    json::write_findings_json(analysis_dir, &findings)?;
    html::write_html_report(analysis_dir, &manifest, &findings)?;
    Ok(())
}
