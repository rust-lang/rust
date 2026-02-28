mod capture;
mod cli;
mod diff;
mod model;
mod provenance;
mod replay;
mod report;
mod scan;
mod util {
    pub mod path_norm;
    pub mod process;
    pub mod redact;
}

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::fs;

use anyhow::{Context, Result};
use camino::Utf8PathBuf;
use clap::Parser;

use crate::capture::{
    CaptureOptions, capture, load_artifacts, load_build_script_messages, load_build_script_stdout,
    load_compiler_artifact_messages, load_invocation_sets, load_run_meta,
};
use crate::cli::{CaptureFlags, Cli, Command, ConfirmLevel, ExplainFlags};
use crate::diff::classify::{classify_artifact, compute_score};
use crate::diff::semantic::{SemanticDiffOptions, build_semantic_diff};
use crate::diff::{load_manifest, run_diff};
use crate::model::{
    AnalysisFindings, BuildScriptExecutedMessage, BuildScriptStdoutRecord, CompilerArtifactMessage,
    DiffClass, DiffStatus, Evidence, Finding, InvocationRecord, ReplayOutcome, SCHEMA_VERSION,
    StageName,
};
use crate::replay::stage::{
    StageLocalization, localize_first_divergent_stage, stage_from_artifact_kind,
};
use crate::replay::{ReplayContext, run_confirmation};
use crate::report::{regenerate_reports, write_reports};
use crate::scan::rules::{fix_hint, primary_locus, top_hits};
use crate::scan::scan_artifact_sources;
use crate::util::path_norm::wildcard_match;

fn main() {
    if let Err(err) = real_main() {
        eprintln!("error: {err:?}");
        std::process::exit(1);
    }
}

fn real_main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::WrapRustc { argv } => {
            let code = capture::run_rustc_wrapper(argv)?;
            std::process::exit(code);
        }
        Command::WrapRustdoc { argv } => {
            let code = capture::run_rustdoc_wrapper(argv)?;
            std::process::exit(code);
        }
        Command::Capture { run_id, keep_target, capture_all_env, binary_dep_depinfo, command } => {
            fs::create_dir_all(&cli.work_dir)
                .with_context(|| format!("failed to create work dir {}", cli.work_dir))?;

            let flags = CaptureFlags { keep_target, capture_all_env, binary_dep_depinfo };
            let res = capture(CaptureOptions {
                run_id,
                work_dir: cli.work_dir,
                command,
                flags,
                extra_env: BTreeMap::new(),
                cwd_override: None,
                target_dir_override: None,
            })?;
            println!(
                "capture complete: run_dir={}, status={}, artifacts={}",
                res.run_dir, res.status, res.artifacts
            );
            if res.status != 0 {
                std::process::exit(res.status);
            }
        }
        Command::Diff { left, right } => {
            fs::create_dir_all(&cli.work_dir)
                .with_context(|| format!("failed to create work dir {}", cli.work_dir))?;
            let diff = run_diff(&cli.work_dir, &left, &right)?;
            println!(
                "diff complete: analysis_dir={}, entries={}",
                diff.analysis_dir,
                diff.manifest.entries.len()
            );
        }
        Command::Explain {
            analysis,
            artifact,
            confirm,
            jobs,
            same_source_replay,
            diffoscope,
            no_diffoscope,
        } => {
            let flags = ExplainFlags {
                artifact_glob: artifact,
                confirm,
                jobs,
                same_source_replay,
                diffoscope,
                no_diffoscope,
            };
            let findings = explain_analysis(&cli.work_dir, &analysis, &flags)?;
            println!(
                "explain complete: analysis_dir={}, findings={}",
                analysis,
                findings.findings.len()
            );
        }
        Command::Report { analysis } => {
            regenerate_reports(&analysis)?;
            println!("report regenerated: {analysis}");
        }
        Command::RunTwice {
            confirm,
            jobs,
            keep_target,
            diffoscope,
            no_diffoscope,
            capture_all_env,
            same_source_replay,
            binary_dep_depinfo,
            command,
        } => {
            fs::create_dir_all(&cli.work_dir)
                .with_context(|| format!("failed to create work dir {}", cli.work_dir))?;

            let capture_flags = CaptureFlags { keep_target, capture_all_env, binary_dep_depinfo };

            let run_a = capture(CaptureOptions {
                run_id: "A".to_string(),
                work_dir: cli.work_dir.clone(),
                command: command.clone(),
                flags: capture_flags.clone(),
                extra_env: BTreeMap::new(),
                cwd_override: None,
                target_dir_override: None,
            })?;
            let run_b = capture(CaptureOptions {
                run_id: "B".to_string(),
                work_dir: cli.work_dir.clone(),
                command: command.clone(),
                flags: capture_flags,
                extra_env: BTreeMap::new(),
                cwd_override: None,
                target_dir_override: None,
            })?;

            let left_run = cli.work_dir.join("runs").join("A");
            let right_run = cli.work_dir.join("runs").join("B");
            let diff_res = run_diff(&cli.work_dir, &left_run, &right_run)?;

            let flags = ExplainFlags {
                artifact_glob: None,
                confirm,
                jobs,
                same_source_replay,
                diffoscope,
                no_diffoscope,
            };
            let findings = explain_analysis(&cli.work_dir, &diff_res.analysis_dir, &flags)?;

            println!(
                "run-twice complete: analysis_dir={}, findings={}, run_status=({}, {})",
                diff_res.analysis_dir,
                findings.findings.len(),
                run_a.status,
                run_b.status,
            );

            if run_a.status != 0 {
                std::process::exit(run_a.status);
            }
            if run_b.status != 0 {
                std::process::exit(run_b.status);
            }
        }
    }

    Ok(())
}

fn explain_analysis(
    work_dir: &Utf8PathBuf,
    analysis_dir: &Utf8PathBuf,
    flags: &ExplainFlags,
) -> Result<AnalysisFindings> {
    let manifest = load_manifest(analysis_dir)?;

    let left_run = Utf8PathBuf::from(&manifest.left_run_dir);
    let right_run = Utf8PathBuf::from(&manifest.right_run_dir);
    let left_meta = load_run_meta(&left_run)?;
    let left_artifacts = load_artifacts(&left_run)?;
    let right_artifacts = load_artifacts(&right_run)?;
    let left_invocations = load_invocation_sets(&left_run)?;
    let right_invocations = load_invocation_sets(&right_run)?;
    let left_compiler_msgs = load_compiler_artifact_messages(&left_run).unwrap_or_default();
    let right_compiler_msgs = load_compiler_artifact_messages(&right_run).unwrap_or_default();
    let left_build_scripts =
        index_build_script_messages(load_build_script_messages(&left_run).unwrap_or_default());
    let right_build_scripts =
        index_build_script_messages(load_build_script_messages(&right_run).unwrap_or_default());
    let left_build_script_stdout =
        index_build_script_stdout(load_build_script_stdout(&left_run).unwrap_or_default());
    let right_build_script_stdout =
        index_build_script_stdout(load_build_script_stdout(&right_run).unwrap_or_default());
    let left_non_fresh_packages = non_fresh_packages(&left_compiler_msgs);
    let right_non_fresh_packages = non_fresh_packages(&right_compiler_msgs);
    let proc_macro_upstream_packages = proc_macro_upstream_packages(&left_run).unwrap_or_default();

    let left_by_id =
        left_artifacts.iter().map(|a| (a.id.clone(), a.clone())).collect::<HashMap<_, _>>();
    let right_by_id =
        right_artifacts.iter().map(|a| (a.id.clone(), a.clone())).collect::<HashMap<_, _>>();
    let left_rustc_by_id = index_invocations_by_id(&left_invocations, "rustc");
    let right_rustc_by_id = index_invocations_by_id(&right_invocations, "rustc");

    let workspace_root = std::path::PathBuf::from(&left_meta.workspace_root);

    let mut replay_cache = HashMap::<String, ReplayOutcome>::new();
    let mut stage_cache = HashMap::<(String, String), StageLocalization>::new();
    let replay_ctx = ReplayContext {
        work_dir: work_dir.clone(),
        command: left_meta.command.clone(),
        command_kind: left_meta.command_kind.clone(),
        capture_flags: CaptureFlags {
            keep_target: false,
            capture_all_env: matches!(left_meta.env_mode, crate::model::EnvMode::All),
            binary_dep_depinfo: false,
        },
        jobs: flags.jobs,
        same_source_replay: flags.same_source_replay,
        focus_package_id: None,
    };

    let mut findings = Vec::new();

    for entry in &manifest.entries {
        if !matches!(
            entry.status,
            DiffStatus::Changed | DiffStatus::LeftOnly | DiffStatus::RightOnly
        ) {
            continue;
        }

        let left_art = entry.left_artifact_id.as_ref().and_then(|id| left_by_id.get(id));
        let right_art = entry.right_artifact_id.as_ref().and_then(|id| right_by_id.get(id));
        let Some(anchor_art) = left_art.or(right_art) else {
            continue;
        };

        if let Some(glob) = &flags.artifact_glob {
            if !wildcard_match(glob, &anchor_art.id)
                && !wildcard_match(glob, &anchor_art.rel_path)
                && !wildcard_match(glob, &anchor_art.path)
            {
                continue;
            }
        }

        let artifact_analysis_dir = analysis_dir.join("artifacts").join(&anchor_art.id);
        fs::create_dir_all(&artifact_analysis_dir)
            .with_context(|| format!("failed to create {artifact_analysis_dir}"))?;

        let semantic =
            if let (Some(l), Some(r)) = (entry.left_path.as_ref(), entry.right_path.as_ref()) {
                build_semantic_diff(
                    std::path::Path::new(l),
                    std::path::Path::new(r),
                    &SemanticDiffOptions {
                        diffoscope: flags.diffoscope.clone(),
                        no_diffoscope: flags.no_diffoscope,
                        left_run_root: Some(left_run.as_std_path().to_path_buf()),
                        right_run_root: Some(right_run.as_std_path().to_path_buf()),
                    },
                )?
            } else {
                crate::model::SemanticDiff {
                    backend: "none".to_string(),
                    summary: "artifact exists only on one side".to_string(),
                    excerpt: "left-only/right-only artifact".to_string(),
                    left_tokens: Vec::new(),
                    right_tokens: Vec::new(),
                }
            };

        write_json(artifact_analysis_dir.join("semantic-diff.json"), &semantic)?;
        fs::write(artifact_analysis_dir.join("semantic-diff.txt"), semantic.excerpt.clone())
            .context("failed to write semantic-diff.txt")?;

        let rule_hits = scan_artifact_sources(anchor_art, &workspace_root);
        write_json(artifact_analysis_dir.join("rule-hits.json"), &rule_hits)?;

        let mut classification = classify_artifact(anchor_art, &semantic, &rule_hits);
        let mut direct_build_script = anchor_art.kind == "out-dir-file"
            || rule_hits.iter().any(|h| h.rule_id == "RE005" || h.rule_id == "RE006");
        if anchor_art
            .package_id
            .as_ref()
            .is_some_and(|pkg| proc_macro_upstream_packages.contains(pkg))
        {
            classification.evidence.push(Evidence {
                kind: "cargo-metadata".to_string(),
                detail: "upstream proc-macro dependency detected for affected package".to_string(),
            });
            if matches!(
                classification.class,
                DiffClass::Unknown
                    | DiffClass::MetadataStage
                    | DiffClass::CodegenStage
                    | DiffClass::LinkStage
            ) {
                classification.class = DiffClass::ProcMacro;
                classification.stage = StageName::ProcMacro;
            }
        }

        if let Some(signal) = build_script_signal(
            anchor_art.package_id.as_ref(),
            &left_build_scripts,
            &right_build_scripts,
            &left_non_fresh_packages,
            &right_non_fresh_packages,
        ) {
            if signal.env_changed {
                if signal.env_order_only {
                    classification.evidence.push(Evidence {
                        kind: "build-script-env".to_string(),
                        detail: "build-script-executed env differs by order only".to_string(),
                    });
                } else {
                    classification.evidence.push(Evidence {
                        kind: "build-script-env".to_string(),
                        detail: "build-script-executed env differs across runs".to_string(),
                    });
                    if matches!(
                        classification.class,
                        DiffClass::Unknown
                            | DiffClass::MetadataStage
                            | DiffClass::CodegenStage
                            | DiffClass::LinkStage
                    ) {
                        classification.class = DiffClass::EnvLeak;
                        classification.stage = StageName::BuildScript;
                    }
                }
            }
            if signal.payload_changed {
                if signal.execution_supported {
                    direct_build_script = true;
                    if signal.order_only {
                        classification.class = DiffClass::UnstableOrder;
                        classification.stage = StageName::BuildScript;
                        classification.evidence.push(Evidence {
                            kind: "build-script-payload".to_string(),
                            detail: "build-script-executed payload differs by order only"
                                .to_string(),
                        });
                    } else {
                        classification.class = DiffClass::BuildScript;
                        classification.stage = StageName::BuildScript;
                        classification.evidence.push(Evidence {
                            kind: "build-script-payload".to_string(),
                            detail: "build-script-executed payload differs across runs".to_string(),
                        });
                    }
                } else {
                    classification.evidence.push(Evidence {
                        kind: "build-script-payload".to_string(),
                        detail: "build-script-executed payload differs, but no nearby non-fresh compiler-artifact evidence for this package"
                            .to_string(),
                    });
                }
            }
        }
        if let Some(stdout_signal) = build_script_stdout_signal(
            anchor_art.package_id.as_ref(),
            &left_build_script_stdout,
            &right_build_script_stdout,
        ) {
            direct_build_script = true;
            if stdout_signal.order_only {
                classification.class = DiffClass::UnstableOrder;
                classification.stage = StageName::BuildScript;
                classification.evidence.push(Evidence {
                    kind: "build-script-stdout".to_string(),
                    detail: "build script cargo:: instruction order differs (order-only)"
                        .to_string(),
                });
            } else {
                if matches!(
                    classification.class,
                    DiffClass::Unknown
                        | DiffClass::MetadataStage
                        | DiffClass::CodegenStage
                        | DiffClass::LinkStage
                        | DiffClass::BuildScript
                ) {
                    classification.class = DiffClass::BuildScript;
                    classification.stage = StageName::BuildScript;
                }
                classification.evidence.push(Evidence {
                    kind: "build-script-stdout".to_string(),
                    detail: "build script cargo:: instruction stream differs".to_string(),
                });
            }
        }

        if matches!(flags.confirm, ConfirmLevel::Standard | ConfirmLevel::Full) {
            if let Some(stage_loc) = stage_localization_for_artifact(
                left_art,
                right_art,
                &left_rustc_by_id,
                &right_rustc_by_id,
                &left_build_scripts,
                &right_build_scripts,
                &artifact_analysis_dir,
                &mut stage_cache,
            ) {
                if write_json(artifact_analysis_dir.join("stage-localization.json"), &stage_loc)
                    .is_ok()
                {
                    classification.evidence.push(Evidence {
                        kind: "stage-replay".to_string(),
                        detail: format!(
                            "first_divergent_stage={}",
                            stage_name_value(&stage_loc.first_divergent_stage)
                        ),
                    });
                }
                if !matches!(stage_loc.first_divergent_stage, StageName::Unknown) {
                    let stage = stage_loc.first_divergent_stage;
                    let keep_proc_macro_stage = classification.class == DiffClass::ProcMacro
                        && !matches!(stage, StageName::BuildScript | StageName::ProcMacro);
                    if !keep_proc_macro_stage {
                        classification.stage = stage;
                    }
                }
            }
        }

        if matches!(classification.stage, crate::model::StageName::Unknown) {
            classification.stage = stage_from_artifact_kind(&anchor_art.kind);
        }

        let replay_outcome = if matches!(flags.confirm, ConfirmLevel::None) {
            None
        } else {
            let cache_key = replay_cache_key(&classification.class, anchor_art.package_id.as_ref());
            if let Some(cached) = replay_cache.get(&cache_key) {
                Some(cached.clone())
            } else {
                let mut replay_ctx = replay_ctx.clone();
                replay_ctx.focus_package_id = anchor_art.package_id.clone();
                let replay = run_confirmation(flags.confirm, &classification.class, &replay_ctx)?;
                if let Some(ref outcome) = replay {
                    replay_cache.insert(cache_key, outcome.clone());
                }
                replay
            }
        };

        let replay_confirmed = replay_outcome.as_ref().is_some_and(|o| o.success);
        if let Some(outcome) = &replay_outcome {
            classification.evidence.push(Evidence {
                kind: "replay".to_string(),
                detail: format!("{}: {}", outcome.experiment, outcome.detail),
            });
        }

        let top_rule_hits = top_hits(&rule_hits, 5);
        for hit in &top_rule_hits {
            classification.evidence.push(Evidence {
                kind: "rule-hit".to_string(),
                detail: format!("{} {}:{} {}", hit.rule_id, hit.path, hit.line, hit.detail),
            });
        }

        let (score, status) = compute_score(
            &classification.class,
            &classification.stage,
            &semantic,
            &rule_hits,
            replay_confirmed,
            direct_build_script,
        );

        let finding = Finding {
            artifact_id: anchor_art.id.clone(),
            status,
            class: classification.class,
            first_divergent_stage: classification.stage,
            primary_locus: primary_locus(&rule_hits),
            evidence: classification.evidence,
            fix_hint: fix_hint(&rule_hits),
            score,
        };

        write_json(artifact_analysis_dir.join("evidence.json"), &finding.evidence)?;
        write_json(artifact_analysis_dir.join("finding.json"), &finding)?;

        findings.push(finding);
    }

    let findings_doc = AnalysisFindings { schema_version: SCHEMA_VERSION, findings };

    write_reports(analysis_dir, &manifest, &findings_doc)?;
    Ok(findings_doc)
}

fn write_json(path: Utf8PathBuf, value: &impl serde::Serialize) -> Result<()> {
    let body = serde_json::to_vec_pretty(value).context("failed to serialize json")?;
    fs::write(&path, body).with_context(|| format!("failed to write {path}"))?;
    Ok(())
}

fn index_invocations_by_id(
    all: &HashMap<String, Vec<InvocationRecord>>,
    tool: &str,
) -> HashMap<String, InvocationRecord> {
    all.get(tool)
        .into_iter()
        .flatten()
        .map(|inv| (inv.id.clone(), inv.clone()))
        .collect::<HashMap<_, _>>()
}

fn stage_localization_for_artifact(
    left_art: Option<&crate::model::ArtifactRecord>,
    right_art: Option<&crate::model::ArtifactRecord>,
    left_by_id: &HashMap<String, InvocationRecord>,
    right_by_id: &HashMap<String, InvocationRecord>,
    left_build_scripts: &HashMap<String, BuildScriptExecutedMessage>,
    right_build_scripts: &HashMap<String, BuildScriptExecutedMessage>,
    artifact_analysis_dir: &Utf8PathBuf,
    cache: &mut HashMap<(String, String), StageLocalization>,
) -> Option<StageLocalization> {
    let (Some(left_art), Some(right_art)) = (left_art, right_art) else {
        return None;
    };
    let (Some(left_inv_id), Some(right_inv_id)) =
        (&left_art.producer_invocation, &right_art.producer_invocation)
    else {
        return None;
    };
    let (Some(left_inv), Some(right_inv)) =
        (left_by_id.get(left_inv_id), right_by_id.get(right_inv_id))
    else {
        return None;
    };

    let key = (left_inv_id.clone(), right_inv_id.clone());
    if let Some(found) = cache.get(&key) {
        return Some(found.clone());
    }

    let package_id = left_art.package_id.as_ref().or(right_art.package_id.as_ref());
    let left_build = package_id.and_then(|pkg| left_build_scripts.get(pkg));
    let right_build = package_id.and_then(|pkg| right_build_scripts.get(pkg));
    let result = localize_first_divergent_stage(
        left_inv,
        right_inv,
        left_build,
        right_build,
        artifact_analysis_dir,
    )
    .ok()?;
    cache.insert(key, result.clone());
    Some(result)
}

fn index_build_script_messages(
    messages: Vec<BuildScriptExecutedMessage>,
) -> HashMap<String, BuildScriptExecutedMessage> {
    messages.into_iter().map(|m| (m.package_id.clone(), m)).collect::<HashMap<_, _>>()
}

fn index_build_script_stdout(
    records: Vec<BuildScriptStdoutRecord>,
) -> HashMap<String, Vec<String>> {
    let mut map = HashMap::<String, Vec<String>>::new();
    for record in records {
        map.entry(record.package_id).or_default().extend(record.lines);
    }
    map
}

#[derive(Debug, Clone, Copy)]
struct BuildScriptSignal {
    payload_changed: bool,
    order_only: bool,
    env_changed: bool,
    env_order_only: bool,
    execution_supported: bool,
}

#[derive(Debug, Clone, Copy)]
struct BuildScriptStdoutSignal {
    order_only: bool,
}

fn build_script_signal(
    package_id: Option<&String>,
    left: &HashMap<String, BuildScriptExecutedMessage>,
    right: &HashMap<String, BuildScriptExecutedMessage>,
    left_non_fresh_packages: &HashSet<String>,
    right_non_fresh_packages: &HashSet<String>,
) -> Option<BuildScriptSignal> {
    let pkg = package_id?;
    let l = left.get(pkg);
    let r = right.get(pkg);
    let execution_supported =
        left_non_fresh_packages.contains(pkg) && right_non_fresh_packages.contains(pkg);

    match (l, r) {
        (Some(l), Some(r)) => {
            if l == r {
                None
            } else {
                Some(BuildScriptSignal {
                    payload_changed: true,
                    order_only: build_script_order_only(l, r),
                    env_changed: l.env != r.env,
                    env_order_only: same_multiset(
                        &l.env.iter().map(|(k, v)| format!("{k}={v}")).collect::<Vec<_>>(),
                        &r.env.iter().map(|(k, v)| format!("{k}={v}")).collect::<Vec<_>>(),
                    ),
                    execution_supported,
                })
            }
        }
        (Some(_), None) | (None, Some(_)) => Some(BuildScriptSignal {
            payload_changed: true,
            order_only: false,
            env_changed: false,
            env_order_only: false,
            execution_supported,
        }),
        (None, None) => None,
    }
}

fn build_script_stdout_signal(
    package_id: Option<&String>,
    left: &HashMap<String, Vec<String>>,
    right: &HashMap<String, Vec<String>>,
) -> Option<BuildScriptStdoutSignal> {
    let pkg = package_id?;
    let l = left.get(pkg);
    let r = right.get(pkg);
    match (l, r) {
        (Some(l), Some(r)) => {
            if l == r {
                None
            } else {
                Some(BuildScriptStdoutSignal { order_only: same_multiset(l, r) })
            }
        }
        (Some(_), None) | (None, Some(_)) => Some(BuildScriptStdoutSignal { order_only: false }),
        (None, None) => None,
    }
}

fn non_fresh_packages(messages: &[CompilerArtifactMessage]) -> HashSet<String> {
    messages
        .iter()
        .filter(|msg| !msg.fresh)
        .map(|msg| msg.package_id.clone())
        .collect::<HashSet<_>>()
}

fn proc_macro_upstream_packages(run_dir: &Utf8PathBuf) -> Result<HashSet<String>> {
    let data = fs::read(run_dir.join("cargo-metadata.json"))
        .with_context(|| format!("failed to read {run_dir}/cargo-metadata.json"))?;
    let metadata: cargo_metadata::Metadata =
        serde_json::from_slice(&data).context("failed to parse cargo-metadata.json")?;

    let proc_macro_packages = metadata
        .packages
        .iter()
        .filter(|pkg| {
            pkg.targets.iter().any(|target| {
                target.kind.iter().any(|k| matches!(k, cargo_metadata::TargetKind::ProcMacro))
            })
        })
        .map(|pkg| pkg.id.to_string())
        .collect::<HashSet<_>>();
    if proc_macro_packages.is_empty() {
        return Ok(HashSet::new());
    }

    let Some(resolve) = metadata.resolve else {
        return Ok(HashSet::new());
    };
    let mut reverse_deps = HashMap::<String, Vec<String>>::new();
    for node in &resolve.nodes {
        let parent = node.id.to_string();
        for dep in &node.deps {
            reverse_deps.entry(dep.pkg.to_string()).or_default().push(parent.clone());
        }
    }

    let mut queue = VecDeque::from_iter(proc_macro_packages);
    let mut upstream = HashSet::<String>::new();
    while let Some(pkg) = queue.pop_front() {
        let Some(parents) = reverse_deps.get(&pkg) else {
            continue;
        };
        for parent in parents {
            if upstream.insert(parent.clone()) {
                queue.push_back(parent.clone());
            }
        }
    }
    Ok(upstream)
}

fn build_script_order_only(
    left: &BuildScriptExecutedMessage,
    right: &BuildScriptExecutedMessage,
) -> bool {
    same_multiset(&left.linked_libs, &right.linked_libs)
        && same_multiset(&left.linked_paths, &right.linked_paths)
        && same_multiset(&left.cfgs, &right.cfgs)
        && same_multiset(
            &left.env.iter().map(|(k, v)| format!("{k}={v}")).collect::<Vec<_>>(),
            &right.env.iter().map(|(k, v)| format!("{k}={v}")).collect::<Vec<_>>(),
        )
}

fn same_multiset(left: &[String], right: &[String]) -> bool {
    if left.len() != right.len() {
        return false;
    }
    let mut counts = HashMap::<&str, i32>::new();
    for item in left {
        *counts.entry(item).or_insert(0) += 1;
    }
    for item in right {
        let Some(v) = counts.get_mut(item.as_str()) else {
            return false;
        };
        *v -= 1;
        if *v < 0 {
            return false;
        }
    }
    counts.values().all(|v| *v == 0)
}

fn stage_name_value(stage: &StageName) -> &'static str {
    match stage {
        StageName::BuildScript => "build-script",
        StageName::ProcMacro => "proc-macro",
        StageName::Metadata => "metadata",
        StageName::Mir => "mir",
        StageName::LlvmIr => "llvm-ir",
        StageName::Obj => "obj",
        StageName::Link => "link",
        StageName::Rustdoc => "rustdoc",
        StageName::Unknown => "unknown",
    }
}

fn replay_cache_key(class: &DiffClass, package_id: Option<&String>) -> String {
    let class_key = match class {
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
    };
    if matches!(class, DiffClass::BuildScript) {
        let pkg = package_id.map(|s| s.as_str()).unwrap_or("<none>");
        format!("{class_key}:{pkg}")
    } else {
        class_key.to_string()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};

    use super::{build_script_signal, build_script_stdout_signal, non_fresh_packages};
    use crate::model::{BuildScriptExecutedMessage, CargoTarget, CompilerArtifactMessage};

    fn build_script_msg(
        package_id: &str,
        linked_libs: &[&str],
        env: &[(&str, &str)],
    ) -> BuildScriptExecutedMessage {
        BuildScriptExecutedMessage {
            reason: "build-script-executed".to_string(),
            package_id: package_id.to_string(),
            linked_libs: linked_libs.iter().map(|v| (*v).to_string()).collect(),
            linked_paths: Vec::new(),
            cfgs: Vec::new(),
            env: env.iter().map(|(k, v)| ((*k).to_string(), (*v).to_string())).collect(),
            out_dir: "/tmp/out".to_string(),
        }
    }

    fn compiler_msg(package_id: &str, fresh: bool) -> CompilerArtifactMessage {
        CompilerArtifactMessage {
            reason: "compiler-artifact".to_string(),
            package_id: package_id.to_string(),
            manifest_path: "/tmp/Cargo.toml".to_string(),
            target: CargoTarget { kind: vec!["lib".to_string()], name: "crate".to_string() },
            profile: None,
            filenames: Vec::new(),
            executable: None,
            fresh,
        }
    }

    #[test]
    fn build_script_signal_requires_non_fresh_support_for_confirmation() {
        let pkg = "file:///tmp/ws#0.1.0".to_string();
        let mut left = HashMap::new();
        let mut right = HashMap::new();
        left.insert(pkg.clone(), build_script_msg(&pkg, &["ssl"], &[]));
        right.insert(pkg.clone(), build_script_msg(&pkg, &["z"], &[]));

        let signal =
            build_script_signal(Some(&pkg), &left, &right, &HashSet::new(), &HashSet::new())
                .expect("signal");
        assert!(signal.payload_changed);
        assert!(!signal.execution_supported);
    }

    #[test]
    fn build_script_signal_detects_env_changes() {
        let pkg = "file:///tmp/ws#0.1.0".to_string();
        let mut left = HashMap::new();
        let mut right = HashMap::new();
        left.insert(pkg.clone(), build_script_msg(&pkg, &["ssl"], &[("A", "1"), ("B", "2")]));
        right.insert(pkg.clone(), build_script_msg(&pkg, &["ssl"], &[("A", "1"), ("B", "3")]));

        let mut non_fresh = HashSet::new();
        non_fresh.insert(pkg.clone());
        let signal =
            build_script_signal(Some(&pkg), &left, &right, &non_fresh, &non_fresh).expect("signal");
        assert!(signal.env_changed);
        assert!(!signal.env_order_only);
        assert!(signal.execution_supported);
    }

    #[test]
    fn non_fresh_package_extraction_ignores_fresh_outputs() {
        let pkgs = non_fresh_packages(&[
            compiler_msg("pkg-a", true),
            compiler_msg("pkg-b", false),
            compiler_msg("pkg-c", false),
        ]);
        assert_eq!(pkgs.len(), 2);
        assert!(pkgs.contains("pkg-b"));
        assert!(pkgs.contains("pkg-c"));
    }

    #[test]
    fn build_script_stdout_signal_detects_order_only() {
        let pkg = "file:///tmp/ws#0.1.0".to_string();
        let mut left = HashMap::new();
        let mut right = HashMap::new();
        left.insert(
            pkg.clone(),
            vec!["cargo::rustc-link-lib=ssl".to_string(), "cargo::rustc-link-lib=z".to_string()],
        );
        right.insert(
            pkg.clone(),
            vec!["cargo::rustc-link-lib=z".to_string(), "cargo::rustc-link-lib=ssl".to_string()],
        );

        let signal = build_script_stdout_signal(Some(&pkg), &left, &right).expect("signal");
        assert!(signal.order_only);
    }
}
