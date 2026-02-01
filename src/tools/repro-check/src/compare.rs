use std::collections::HashSet;
use std::fs::File;
use std::io::copy;
use std::path::{Path, PathBuf};

use anyhow::Result;
use chrono::Utc;
use log::{info, trace, warn};
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use walkdir::{DirEntry, WalkDir};

#[derive(Debug, Clone)]
pub struct ComparisonReport {
    pub mismatches: Vec<Mismatch>,
    pub total_files: usize,
    pub matching_files: usize,
    pub ignored_files: Vec<(PathBuf, String)>,
    pub compared_files: Vec<PathBuf>,
}

#[derive(Debug, Clone)]
pub struct Mismatch {
    pub path: PathBuf,
    pub hash_a: String,
    pub hash_b: String,
}

/// Compares two directories, ignoring certain file patterns.
/// Collects files from dir_a, filters them, hashes in parallel, then checks against dir_b.
/// We sort entries for consistent ordering - helps with debugging.
pub fn compare_directories(
    dir_a: &Path,
    dir_b: &Path,
    host: &str,
    exclude_patterns: &HashSet<String>,
) -> Result<ComparisonReport> {
    let mut entries_a: Vec<DirEntry> = WalkDir::new(dir_a)
        .sort_by_file_name()
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .collect();

    let mut ignored_files = Vec::new();
    let mut compared_files = Vec::new();

    entries_a.retain(|entry| {
        let fname = entry.file_name().to_string_lossy().to_string();

        // Always compare lowercase for case-insensitive suffix match
        let name_to_check = fname.to_lowercase();

        for pat in exclude_patterns {
            let pat_to_check = pat.to_lowercase();

            if name_to_check.ends_with(&pat_to_check) {
                let rel = entry.path().strip_prefix(dir_a).unwrap().to_path_buf();
                ignored_files.push((rel, pat.clone()));
                return false;
            }
        }

        let rel = entry.path().strip_prefix(dir_a).unwrap().to_path_buf();
        compared_files.push(rel);
        true
    });

    let total_files = entries_a.len() + ignored_files.len();
    trace!("Found {} files to compare, ignored {}", entries_a.len(), ignored_files.len());

    let hashes_a: Vec<(PathBuf, String)> = entries_a
        .par_iter()
        .map(|entry| {
            let rel_path = entry.path().strip_prefix(dir_a).unwrap().to_path_buf();
            match compute_hash(entry.path()) {
                Ok(h) => (rel_path, h),
                Err(e) => {
                    warn!("Hash error on {:?}: {}", entry.path(), e);
                    (rel_path, "HASH_ERROR".to_string())
                }
            }
        })
        .collect();

    let mut mismatches = Vec::new();
    for (rel_path, hash_a) in hashes_a {
        let path_b = dir_b.join(&rel_path);
        let hash_b = if path_b.exists() {
            compute_hash(&path_b)
                .map_err(|e| warn!("Hash fail on B {:?}: {}", path_b, e))
                .unwrap_or("HASH_ERROR".to_string())
        } else {
            "MISSING_FILE".to_string()
        };

        if hash_a != hash_b {
            mismatches.push(Mismatch { path: rel_path, hash_a, hash_b });
        }
    }

    let matching_files = compared_files.len() - mismatches.len();
    info!("Compared on host {} - mismatches: {}", host, mismatches.len());

    Ok(ComparisonReport { mismatches, total_files, matching_files, ignored_files, compared_files })
}

/// Builds an HTML report from the comparison results.
pub fn generate_html_report(report: &ComparisonReport, output_path: &Path) -> Result<()> {
    let (status_class, status_text) =
        if report.mismatches.is_empty() { ("success", "PASSED") } else { ("failure", "FAILED") };

    let timestamp = Utc::now().format("%Y-%m-%d %H:%M:%S UTC").to_string();
    let mut html = String::new();

    html.push_str(&format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>Repro Check Report</title>
    <style>
        body {{ font-family: monospace; margin: 2rem; background: #f8f9fa; }}
        .container {{ max-width: 80rem; margin: auto; }}
        .header {{ padding: 1.5rem; background: #e9ecef; border-radius: 0.5rem; margin-bottom: 1.5rem; }}
        h1 {{ margin: 0; font-size: 1.8rem; }}
        .success h1 {{ color: green; }}
        .failure h1 {{ color: red; }}
        .summary {{ font-size: 1rem; margin: 0.75rem 0; }}
        table {{ width: 100%; border-collapse: collapse; margin: 1.25rem 0; background: white; }}
        th, td {{ border: 1px solid #dee2e6; padding: 0.625rem; text-align: left; }}
        th {{ background: #f8f9fa; }}
        .mismatch {{ background: #ffe5e5; }}
        .ignored {{ background: #fff3cd; }}
        .section {{ margin: 2rem 0; }}
        .count {{ padding: 0.25rem 0.5rem; border-radius: 0.25rem; font-weight: bold; }}
        .count.match {{ background: #d4edda; color: green; }}
        .count.mismatch {{ background: #f8d7da; color: red; }}
        .count.ignored {{ background: #fff3cd; color: orange; }}
        details {{ margin: 1rem 0; }}
        summary {{ cursor: pointer; font-weight: bold; }}
    </style>
</head>
<body>
<div class="container">
    <div class="header {status_class}">
        <h1>Repro Check: {status_text}</h1>
        <div class="summary">
            <strong>Total files:</strong> {total} |
            <span class="count match">Matching: {matching}</span> |
            <span class="count mismatch">Mismatches: {mcount}</span> |
            <span class="count ignored">Ignored: {icount}</span>
        </div>
    </div>

    <div class="section">
        <h2>Mismatches ({mcount})</h2>"#,
        status_class = status_class,
        status_text = status_text,
        total = report.total_files,
        matching = report.matching_files,
        mcount = report.mismatches.len(),
        icount = report.ignored_files.len(),
    ));

    if report.mismatches.is_empty() {
        html.push_str("<p>Everything matches - good job!</p>");
    } else {
        html.push_str(r#"
            <table>
                <thead><tr><th>File Path</th><th>Hash A (short)</th><th>Hash B (short)</th></tr></thead>
                <tbody>
        "#);
        for mismatch in &report.mismatches {
            let short_a = mismatch.hash_a.get(..16).unwrap_or("N/A");
            let short_b = mismatch.hash_b.get(..16).unwrap_or("N/A");
            html.push_str(&format!(
                r#"<tr class="mismatch"><td>{}</td><td>{}</td><td>{}</td></tr>"#,
                mismatch.path.display(),
                short_a,
                short_b
            ));
        }
        html.push_str("</tbody></table>");
    }

    html.push_str(&format!(
        r#"
    </div>
    <div class="section">
        <h2>Ignored Files ({})</h2>"#,
        report.ignored_files.len()
    ));

    if report.ignored_files.is_empty() {
        html.push_str("<p>None ignored this time.</p>");
    } else {
        html.push_str(
            r#"
            <details open>
                <summary>Click to hide/show</summary>
                <table>
                    <thead><tr><th>File</th><th>Matched Pattern</th></tr></thead>
                    <tbody>
        "#,
        );
        for (path, pat) in &report.ignored_files {
            html.push_str(&format!(
                r#"<tr class="ignored"><td>{}</td><td>{}</td></tr>"#,
                path.display(),
                pat
            ));
        }
        html.push_str("</tbody></table></details>");
    }

    html.push_str(&format!(
        r#"
    </div>
    <div class="section">
        <h2>Files Compared ({})</h2>"#,
        report.compared_files.len()
    ));

    if report.compared_files.is_empty() {
        html.push_str("<p>Nothing to compare - maybe all ignored?</p>");
    } else {
        html.push_str(
            r#"
            <details>
                <summary>Expand to see list</summary>
                <ul>
        "#,
        );
        for path in &report.compared_files {
            html.push_str(&format!("<li>{}</li>", path.display()));
        }
        html.push_str("</ul></details>");
    }

    html.push_str(&format!(
        r#"
    </div>
    <footer style="margin-top: 3rem; color: #6c757d; font-size: 0.875rem; text-align: center;">
        Report generated on {timestamp}
    </footer>
</div>
</body>
</html>"#,
        timestamp = timestamp
    ));

    std::fs::write(output_path, html)?;
    info!("Wrote report to {}", output_path.display());
    Ok(())
}

/// Simple hash func - SHA256, copies file content into hasher.
pub fn compute_hash(path: &Path) -> Result<String> {
    let mut f = File::open(path)?;
    let mut hasher = Sha256::new();
    copy(&mut f, &mut hasher)?;
    Ok(hex::encode(hasher.finalize()))
}
