use std::collections::{BTreeMap, HashMap};
use std::fs;

use anyhow::{Context, Result};
use camino::Utf8PathBuf;

use crate::capture::{load_invocation_sets, load_run_meta};
use crate::model::{
    AnalysisFindings, DiffClass, DiffEntry, DiffManifest, Finding, FindingStatus, InvocationRecord,
    RunMeta, SourceLocus,
};

pub fn write_html_report(
    analysis_dir: &Utf8PathBuf,
    manifest: &DiffManifest,
    findings: &AnalysisFindings,
) -> Result<()> {
    let mut status_counts = BTreeMap::<String, usize>::new();
    let mut class_counts = BTreeMap::<String, usize>::new();
    let entry_by_artifact = build_entry_index(manifest);
    let run_details = load_run_details(manifest);

    let mut confirmed_items = Vec::<String>::new();
    let mut rows = String::new();
    for finding in &findings.findings {
        *status_counts
            .entry(
                match finding.status {
                    FindingStatus::Confirmed => "confirmed",
                    FindingStatus::StrongSuspect => "strong-suspect",
                    FindingStatus::WeakSuspect => "weak-suspect",
                }
                .to_string(),
            )
            .or_default() += 1;

        *class_counts.entry(class_name(&finding.class).to_string()).or_default() += 1;

        let detail_rel = write_artifact_detail_page(
            analysis_dir,
            finding,
            entry_by_artifact.get(&finding.artifact_id).copied(),
            &run_details.run_roots,
            &run_details.workspace_roots,
        )?;
        let locus = finding
            .primary_locus
            .as_ref()
            .map(|l| {
                format!(
                    "{}:{}",
                    html_escape(&redact_for_report(
                        &l.path,
                        &run_details.run_roots,
                        &run_details.workspace_roots,
                    )),
                    l.line
                )
            })
            .unwrap_or_else(|| "-".to_string());

        rows.push_str(&format!(
            "<tr><td><a href=\"{}\">{}</a></td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>",
            html_escape(&detail_rel),
            html_escape(&finding.artifact_id),
            html_escape(class_name(&finding.class)),
            html_escape(status_name(&finding.status)),
            html_escape(stage_name(&finding.first_divergent_stage)),
            locus
        ));

        if matches!(finding.status, FindingStatus::Confirmed) {
            confirmed_items.push(format!(
                "<li><a href=\"{}\">{}</a> <small>{}</small></li>",
                html_escape(&detail_rel),
                html_escape(&finding.artifact_id),
                html_escape(class_name(&finding.class))
            ));
        }
    }

    let status_html = status_counts
        .iter()
        .map(|(k, v)| format!("<li><b>{}</b>: {}</li>", html_escape(k), v))
        .collect::<Vec<_>>()
        .join("\n");
    let class_html = class_counts
        .iter()
        .map(|(k, v)| format!("<li><b>{}</b>: {}</li>", html_escape(k), v))
        .collect::<Vec<_>>()
        .join("\n");

    let build_script_diff = findings.findings.iter().any(|f| {
        f.class == DiffClass::BuildScript
            || f.evidence.iter().any(|e| e.kind == "build-script-payload")
    });

    let confirmed_html = if confirmed_items.is_empty() {
        "<li>(none)</li>".to_string()
    } else {
        confirmed_items.join("\n")
    };
    let env_diff_html = if run_details.env_diff.is_empty() {
        "<li>(none)</li>".to_string()
    } else {
        run_details
            .env_diff
            .iter()
            .map(|line| format!("<li><code>{}</code></li>", html_escape(line)))
            .collect::<Vec<_>>()
            .join("\n")
    };

    let html = format!(
        r#"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>repro-explain report</title>
<style>
:root {{
  --bg: #f5f3ec;
  --panel: #fffdf7;
  --ink: #1f2a2a;
  --muted: #516061;
  --accent: #c44a2b;
  --line: #d7d2c7;
}}
body {{ margin: 0; font-family: "Iowan Old Style", "Palatino Linotype", serif; background: linear-gradient(180deg, #faf7ef, #f0ece2); color: var(--ink); }}
main {{ max-width: 1100px; margin: 0 auto; padding: 24px; }}
header {{ border-bottom: 2px solid var(--line); margin-bottom: 20px; }}
.card {{ background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 14px 16px; margin-bottom: 14px; box-shadow: 0 4px 16px rgba(0,0,0,0.04); }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 12px; }}
table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
th, td {{ border-bottom: 1px solid var(--line); padding: 8px; text-align: left; vertical-align: top; }}
th {{ color: var(--muted); font-size: 12px; letter-spacing: .04em; text-transform: uppercase; }}
.badge {{ color: var(--panel); background: var(--accent); border-radius: 999px; padding: 2px 8px; font-size: 12px; }}
small {{ color: var(--muted); }}
</style>
</head>
<body>
<main>
<header>
  <h1>repro-explain</h1>
  <p><small>left run: {left}<br>right run: {right}<br>left command: {left_cmd}<br>right command: {right_cmd}<br>env mode: {left_env_mode} / {right_env_mode}<br>artifact pairs: {pairs}<br>build script diff: {build_script_diff}</small></p>
</header>
<section class="grid">
  <div class="card"><h2>Status</h2><ul>{status_html}</ul></div>
  <div class="card"><h2>Class</h2><ul>{class_html}</ul></div>
  <div class="card"><h2>Confirmed</h2><ul>{confirmed_html}</ul></div>
  <div class="card"><h2>Environment Diff</h2><ul>{env_diff_html}</ul></div>
</section>
<section class="card">
  <h2>Findings <span class="badge">{finding_count}</span></h2>
  <table>
    <thead><tr><th>Artifact</th><th>Class</th><th>Status</th><th>First Stage</th><th>Primary Locus</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
</section>
</main>
</body>
</html>
"#,
        left = html_escape(&redact_for_report(
            &manifest.left_run_dir,
            &run_details.run_roots,
            &run_details.workspace_roots,
        )),
        right = html_escape(&redact_for_report(
            &manifest.right_run_dir,
            &run_details.run_roots,
            &run_details.workspace_roots,
        )),
        left_cmd = html_escape(&redact_for_report(
            &run_details.left_command,
            &run_details.run_roots,
            &run_details.workspace_roots,
        )),
        right_cmd = html_escape(&redact_for_report(
            &run_details.right_command,
            &run_details.run_roots,
            &run_details.workspace_roots,
        )),
        left_env_mode = html_escape(&run_details.left_env_mode),
        right_env_mode = html_escape(&run_details.right_env_mode),
        pairs = manifest.entries.len(),
        build_script_diff = if build_script_diff { "yes" } else { "no" },
        status_html = status_html,
        class_html = class_html,
        confirmed_html = confirmed_html,
        env_diff_html = env_diff_html,
        finding_count = findings.findings.len(),
        rows = rows,
    );

    fs::write(analysis_dir.join("report.html"), html)
        .with_context(|| format!("failed to write {analysis_dir}/report.html"))?;
    Ok(())
}

#[derive(Debug, Default)]
struct RunDetails {
    left_command: String,
    right_command: String,
    left_env_mode: String,
    right_env_mode: String,
    env_diff: Vec<String>,
    run_roots: Vec<String>,
    workspace_roots: Vec<String>,
}

fn load_run_details(manifest: &DiffManifest) -> RunDetails {
    let left_run = Utf8PathBuf::from(&manifest.left_run_dir);
    let right_run = Utf8PathBuf::from(&manifest.right_run_dir);

    let left_meta = load_run_meta(&left_run).ok();
    let right_meta = load_run_meta(&right_run).ok();
    let left_invocations = load_invocation_sets(&left_run).ok();
    let right_invocations = load_invocation_sets(&right_run).ok();

    let left_env = left_invocations.as_ref().and_then(first_env_snapshot).unwrap_or_default();
    let right_env = right_invocations.as_ref().and_then(first_env_snapshot).unwrap_or_default();
    let workspace_roots = collect_workspace_roots(left_meta.as_ref(), right_meta.as_ref());
    let run_roots = collect_run_roots(manifest);

    RunDetails {
        left_command: left_meta
            .as_ref()
            .map(command_from_meta)
            .unwrap_or_else(|| "(unavailable)".to_string()),
        right_command: right_meta
            .as_ref()
            .map(command_from_meta)
            .unwrap_or_else(|| "(unavailable)".to_string()),
        left_env_mode: left_meta
            .as_ref()
            .map(env_mode_from_meta)
            .unwrap_or_else(|| "(unavailable)".to_string()),
        right_env_mode: right_meta
            .as_ref()
            .map(env_mode_from_meta)
            .unwrap_or_else(|| "(unavailable)".to_string()),
        env_diff: diff_env_snapshot(&left_env, &right_env, &run_roots, &workspace_roots),
        run_roots,
        workspace_roots,
    }
}

fn command_from_meta(meta: &RunMeta) -> String {
    if meta.command.is_empty() { "(empty)".to_string() } else { meta.command.join(" ") }
}

fn env_mode_from_meta(meta: &RunMeta) -> String {
    match meta.env_mode {
        crate::model::EnvMode::Allowlist => "allowlist".to_string(),
        crate::model::EnvMode::All => "all".to_string(),
    }
}

fn first_env_snapshot(
    invocations: &HashMap<String, Vec<InvocationRecord>>,
) -> Option<BTreeMap<String, String>> {
    for tool in ["rustc", "rustdoc"] {
        if let Some(records) = invocations.get(tool) {
            for record in records {
                if !record.env.is_empty() {
                    return Some(record.env.clone());
                }
            }
        }
    }
    for records in invocations.values() {
        for record in records {
            if !record.env.is_empty() {
                return Some(record.env.clone());
            }
        }
    }
    None
}

fn diff_env_snapshot(
    left_env: &BTreeMap<String, String>,
    right_env: &BTreeMap<String, String>,
    run_roots: &[String],
    workspace_roots: &[String],
) -> Vec<String> {
    let keys =
        left_env.keys().chain(right_env.keys()).cloned().collect::<std::collections::BTreeSet<_>>();
    if keys.is_empty() {
        return Vec::new();
    }

    let mut out = Vec::new();
    for key in keys {
        let left = left_env.get(&key);
        let right = right_env.get(&key);
        if left == right {
            continue;
        }
        let left_value = left.map_or("<unset>", String::as_str);
        let right_value = right.map_or("<unset>", String::as_str);
        out.push(format!(
            "{}: {} -> {}",
            key,
            truncate_value(&redact_for_report(left_value, run_roots, workspace_roots)),
            truncate_value(&redact_for_report(right_value, run_roots, workspace_roots))
        ));
        if out.len() >= 20 {
            break;
        }
    }
    out
}

fn truncate_value(value: &str) -> String {
    const LIMIT: usize = 80;
    if value.chars().count() <= LIMIT {
        value.to_string()
    } else {
        format!("{}...", value.chars().take(LIMIT).collect::<String>())
    }
}

fn write_artifact_detail_page(
    analysis_dir: &Utf8PathBuf,
    finding: &Finding,
    entry: Option<&DiffEntry>,
    run_roots: &[String],
    workspace_roots: &[String],
) -> Result<String> {
    let artifact_dir = analysis_dir.join("artifacts").join(&finding.artifact_id);
    fs::create_dir_all(&artifact_dir)
        .with_context(|| format!("failed to create artifact detail dir {}", artifact_dir))?;

    let semantic_diff = fs::read_to_string(artifact_dir.join("semantic-diff.txt"))
        .unwrap_or_else(|_| "(semantic diff unavailable)".to_string());
    let semantic_excerpt = truncate_lines(&semantic_diff, 240);

    let stage_localization = fs::read_to_string(artifact_dir.join("stage-localization.json"))
        .unwrap_or_else(|_| "(stage localization unavailable)".to_string());
    let stage_replay_excerpt = truncate_lines(&stage_localization, 80);

    let top_findings = finding
        .evidence
        .iter()
        .take(5)
        .map(|e| format!("<li><b>{}</b>: {}</li>", html_escape(&e.kind), html_escape(&e.detail)))
        .collect::<Vec<_>>()
        .join("\n");

    let snippet = source_snippet_html(finding.primary_locus.as_ref())
        .unwrap_or_else(|| "<pre><code>(source snippet unavailable)</code></pre>".to_string());

    let replay_html = finding
        .evidence
        .iter()
        .filter(|e| e.kind == "replay" || e.kind == "stage-replay")
        .map(|e| format!("<li>{}</li>", html_escape(&e.detail)))
        .collect::<Vec<_>>();
    let replay_html =
        if replay_html.is_empty() { "<li>(none)</li>".to_string() } else { replay_html.join("\n") };

    let (left_sha, right_sha, status, kind) = if let Some(entry) = entry {
        (
            entry.left_sha256.clone().unwrap_or_else(|| "-".to_string()),
            entry.right_sha256.clone().unwrap_or_else(|| "-".to_string()),
            format!("{:?}", entry.status).to_lowercase(),
            entry.kind.clone(),
        )
    } else {
        ("-".to_string(), "-".to_string(), "-".to_string(), "-".to_string())
    };

    let fix_hint = finding.fix_hint.clone().unwrap_or_else(|| "(none)".to_string());
    let locus = finding
        .primary_locus
        .as_ref()
        .map(|l| {
            format!(
                "{}:{}",
                html_escape(&redact_for_report(&l.path, run_roots, workspace_roots)),
                l.line
            )
        })
        .unwrap_or_else(|| "-".to_string());

    let html = format!(
        r#"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>artifact {artifact}</title>
<style>
body {{ margin: 0; font-family: "Iowan Old Style", "Palatino Linotype", serif; background: #f6f3ea; color: #1f2a2a; }}
main {{ max-width: 980px; margin: 0 auto; padding: 20px; }}
.card {{ background: #fffdf7; border: 1px solid #d7d2c7; border-radius: 10px; padding: 12px 14px; margin-bottom: 12px; }}
pre {{ white-space: pre-wrap; overflow-wrap: anywhere; background: #f4efe4; border: 1px solid #ddd1be; border-radius: 6px; padding: 10px; }}
code {{ font-family: "Fira Code", "Consolas", monospace; font-size: 12px; }}
</style>
</head>
<body>
<main>
<div class="card"><a href="../../report.html">Back to summary</a></div>
<section class="card"><h2>1. Artifact</h2><p>id: {artifact}<br>class: {class}<br>status: {status_name}<br>kind: {kind}<br>primary locus: {locus}</p></section>
<section class="card"><h2>2. Byte Hash Diff</h2><p>left sha256: {left_sha}<br>right sha256: {right_sha}<br>manifest status: {manifest_status}</p></section>
<section class="card"><h2>3. Semantic Diff</h2><pre><code>{semantic}</code></pre></section>
<section class="card"><h2>4. First Divergent Stage</h2><p>{stage}</p><pre><code>{stage_replay}</code></pre></section>
<section class="card"><h2>5. Top Findings</h2><ul>{top_findings}</ul></section>
<section class="card"><h2>6. Source Snippets</h2>{snippet}</section>
<section class="card"><h2>7. Replay Experiments</h2><ul>{replay}</ul></section>
<section class="card"><h2>8. Suggested Fixes</h2><p>{fix_hint}</p></section>
</main>
</body>
</html>
"#,
        artifact = html_escape(&finding.artifact_id),
        class = html_escape(class_name(&finding.class)),
        status_name = html_escape(status_name(&finding.status)),
        kind = html_escape(&kind),
        locus = locus,
        left_sha = html_escape(&left_sha),
        right_sha = html_escape(&right_sha),
        manifest_status = html_escape(&status),
        semantic = html_escape(&semantic_excerpt),
        stage = html_escape(stage_name(&finding.first_divergent_stage)),
        stage_replay = html_escape(&stage_replay_excerpt),
        top_findings =
            if top_findings.is_empty() { "<li>(none)</li>".to_string() } else { top_findings },
        snippet = snippet,
        replay = replay_html,
        fix_hint = html_escape(&fix_hint),
    );

    fs::write(artifact_dir.join("index.html"), html)
        .with_context(|| format!("failed to write detail page for {}", finding.artifact_id))?;

    Ok(format!("artifacts/{}/index.html", finding.artifact_id))
}

fn collect_workspace_roots(
    left_meta: Option<&RunMeta>,
    right_meta: Option<&RunMeta>,
) -> Vec<String> {
    let mut roots = Vec::new();
    if let Some(meta) = left_meta {
        roots.push(meta.workspace_root.clone());
    }
    if let Some(meta) = right_meta {
        roots.push(meta.workspace_root.clone());
    }
    normalize_roots(roots)
}

fn collect_run_roots(manifest: &DiffManifest) -> Vec<String> {
    normalize_roots(vec![manifest.left_run_dir.clone(), manifest.right_run_dir.clone()])
}

fn normalize_roots(mut roots: Vec<String>) -> Vec<String> {
    roots.retain(|r| !r.is_empty());
    roots.sort_by_key(|r| std::cmp::Reverse(r.len()));
    roots.dedup();
    roots
}

fn redact_for_report(value: &str, run_roots: &[String], workspace_roots: &[String]) -> String {
    let mut redacted = value.to_string();
    for root in run_roots {
        if !root.is_empty() {
            redacted = redacted.replace(root, "<run-root>");
        }
    }
    for root in workspace_roots {
        if !root.is_empty() {
            redacted = redacted.replace(root, "<workspace-root>");
        }
    }
    redacted
}

fn source_snippet_html(locus: Option<&SourceLocus>) -> Option<String> {
    let locus = locus?;
    let body = fs::read_to_string(&locus.path).ok()?;
    let lines = body.lines().collect::<Vec<_>>();
    if lines.is_empty() {
        return None;
    }
    let line = locus.line.saturating_sub(1);
    let from = line.saturating_sub(2);
    let to = usize::min(lines.len(), line + 3);
    let mut out = String::new();
    for idx in from..to {
        out.push_str(&format!("{:>6} | {}\n", idx + 1, lines[idx]));
    }
    Some(format!("<pre><code>{}</code></pre>", html_escape(&out)))
}

fn build_entry_index<'a>(manifest: &'a DiffManifest) -> HashMap<String, &'a DiffEntry> {
    let mut map = HashMap::new();
    for entry in &manifest.entries {
        if let Some(left) = &entry.left_artifact_id {
            map.insert(left.clone(), entry);
        }
        if let Some(right) = &entry.right_artifact_id {
            map.insert(right.clone(), entry);
        }
    }
    map
}

fn truncate_lines(text: &str, max_lines: usize) -> String {
    let mut out = String::new();
    for (i, line) in text.lines().enumerate() {
        if i >= max_lines {
            out.push_str("... truncated ...\n");
            break;
        }
        out.push_str(line);
        out.push('\n');
    }
    if out.is_empty() { "(empty)".to_string() } else { out }
}

fn html_escape(input: &str) -> String {
    input
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

fn status_name(status: &FindingStatus) -> &'static str {
    match status {
        FindingStatus::Confirmed => "confirmed",
        FindingStatus::StrongSuspect => "strong-suspect",
        FindingStatus::WeakSuspect => "weak-suspect",
    }
}

fn class_name(class: &DiffClass) -> &'static str {
    match class {
        DiffClass::PathLeak => "path-leak",
        DiffClass::Timestamp => "timestamp",
        DiffClass::EnvLeak => "env-leak",
        DiffClass::UnstableOrder => "unstable-order",
        DiffClass::BuildScript => "build-script",
        DiffClass::ProcMacro => "proc-macro",
        DiffClass::MetadataStage => "metadata-stage",
        DiffClass::CodegenStage => "codegen-stage",
        DiffClass::LinkStage => "link-stage",
        DiffClass::ScheduleSensitiveParallelism => "schedule-sensitive-parallelism",
        DiffClass::Unknown => "unknown",
    }
}

fn stage_name(stage: &crate::model::StageName) -> &'static str {
    match stage {
        crate::model::StageName::BuildScript => "build-script",
        crate::model::StageName::ProcMacro => "proc-macro",
        crate::model::StageName::Metadata => "metadata",
        crate::model::StageName::Mir => "mir",
        crate::model::StageName::LlvmIr => "llvm-ir",
        crate::model::StageName::Obj => "obj",
        crate::model::StageName::Link => "link",
        crate::model::StageName::Rustdoc => "rustdoc",
        crate::model::StageName::Unknown => "unknown",
    }
}

#[cfg(test)]
#[path = "tests/html.rs"]
mod tests;
