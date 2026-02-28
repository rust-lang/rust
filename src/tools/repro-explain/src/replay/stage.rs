use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result};
use camino::Utf8PathBuf;
use serde::{Deserialize, Serialize};

use crate::capture::hash::sha256_file;
use crate::model::{BuildScriptExecutedMessage, InvocationRecord, StageName};

pub fn stage_from_artifact_kind(kind: &str) -> StageName {
    match kind {
        "out-dir-file" => StageName::BuildScript,
        "rmeta" => StageName::Metadata,
        "llvm-ir" | "llvm-bc" => StageName::LlvmIr,
        "obj" => StageName::Obj,
        "rustdoc-html" => StageName::Rustdoc,
        "dylib" | "rlib" | "staticlib" | "binary" => StageName::Link,
        _ => StageName::Unknown,
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageCheck {
    pub stage: StageName,
    pub left_status: i32,
    pub right_status: i32,
    pub left_path: Option<String>,
    pub right_path: Option<String>,
    pub left_sha256: Option<String>,
    pub right_sha256: Option<String>,
    pub equal: bool,
    pub detail: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageLocalization {
    pub first_divergent_stage: StageName,
    pub checks: Vec<StageCheck>,
}

pub fn localize_first_divergent_stage(
    left: &InvocationRecord,
    right: &InvocationRecord,
    left_build_script: Option<&BuildScriptExecutedMessage>,
    right_build_script: Option<&BuildScriptExecutedMessage>,
    scratch_root: &Utf8PathBuf,
) -> Result<StageLocalization> {
    let mut checks = Vec::new();
    if let Some(build_script_check) =
        build_script_stage_check(left_build_script, right_build_script)
    {
        let diverged = !build_script_check.equal;
        checks.push(build_script_check);
        if diverged {
            return Ok(StageLocalization { first_divergent_stage: StageName::BuildScript, checks });
        }
    }

    if left.tool != "rustc" || right.tool != "rustc" {
        return Ok(StageLocalization { first_divergent_stage: StageName::Unknown, checks });
    }

    let is_proc_macro_invocation = left.crate_types.iter().any(|ty| ty == "proc-macro")
        || right.crate_types.iter().any(|ty| ty == "proc-macro");

    let stages = [
        (StageName::Metadata, "metadata", "rmeta"),
        (StageName::Mir, "mir", "mir"),
        (StageName::LlvmIr, "llvm-ir", "ll"),
        (StageName::Obj, "obj", "o"),
    ];

    std::fs::create_dir_all(scratch_root.join("replay"))
        .with_context(|| format!("failed to create replay dir under {scratch_root}"))?;
    let temp = tempfile::Builder::new()
        .prefix("stage-")
        .tempdir_in(scratch_root.join("replay").as_std_path())
        .context("failed to create stage replay temp dir")?;

    let mut first = StageName::Link;

    for (stage, emit, ext) in stages {
        let left_side = run_single_stage(
            left,
            "left",
            emit,
            ext,
            left_build_script.map(|msg| msg.env.as_slice()),
            temp.path(),
        )?;
        let right_side = run_single_stage(
            right,
            "right",
            emit,
            ext,
            right_build_script.map(|msg| msg.env.as_slice()),
            temp.path(),
        )?;

        let equal = left_side.sha256.is_some()
            && right_side.sha256.is_some()
            && left_side.sha256 == right_side.sha256;
        let detail = format!(
            "left_status={}, right_status={}, left_stderr={}, right_stderr={}",
            left_side.status,
            right_side.status,
            left_side.stderr_excerpt,
            right_side.stderr_excerpt
        );

        checks.push(StageCheck {
            stage: stage.clone(),
            left_status: left_side.status,
            right_status: right_side.status,
            left_path: left_side.path.map(|p| p.to_string_lossy().into_owned()),
            right_path: right_side.path.map(|p| p.to_string_lossy().into_owned()),
            left_sha256: left_side.sha256.clone(),
            right_sha256: right_side.sha256.clone(),
            equal,
            detail,
        });

        if left_side.status != 0 || right_side.status != 0 {
            first = StageName::Unknown;
            break;
        }
        if !equal {
            first = stage;
            break;
        }
    }

    if checks.is_empty() {
        first = StageName::Unknown;
    }
    first = map_proc_macro_stage(first, is_proc_macro_invocation);

    Ok(StageLocalization { first_divergent_stage: first, checks })
}

fn build_script_stage_check(
    left: Option<&BuildScriptExecutedMessage>,
    right: Option<&BuildScriptExecutedMessage>,
) -> Option<StageCheck> {
    let (left_status, right_status) =
        (if left.is_some() { 0 } else { 1 }, if right.is_some() { 0 } else { 1 });
    let (left_path, right_path) =
        (left.map(|m| m.out_dir.clone()), right.map(|m| m.out_dir.clone()));

    match (left, right) {
        (None, None) => None,
        (Some(l), Some(r)) if l == r => Some(StageCheck {
            stage: StageName::BuildScript,
            left_status,
            right_status,
            left_path,
            right_path,
            left_sha256: None,
            right_sha256: None,
            equal: true,
            detail: "build-script-executed payload is identical".to_string(),
        }),
        (Some(l), Some(r)) => {
            let env_left = l.env.iter().map(|(k, v)| format!("{k}={v}")).collect::<Vec<_>>();
            let env_right = r.env.iter().map(|(k, v)| format!("{k}={v}")).collect::<Vec<_>>();
            let order_only = same_multiset(&l.linked_libs, &r.linked_libs)
                && same_multiset(&l.linked_paths, &r.linked_paths)
                && same_multiset(&l.cfgs, &r.cfgs)
                && same_multiset(&env_left, &env_right);
            Some(StageCheck {
                stage: StageName::BuildScript,
                left_status,
                right_status,
                left_path,
                right_path,
                left_sha256: None,
                right_sha256: None,
                equal: false,
                detail: format!("build-script-executed payload differs (order_only={order_only})"),
            })
        }
        (Some(_), None) | (None, Some(_)) => Some(StageCheck {
            stage: StageName::BuildScript,
            left_status,
            right_status,
            left_path,
            right_path,
            left_sha256: None,
            right_sha256: None,
            equal: false,
            detail: "build-script-executed payload exists only on one side".to_string(),
        }),
    }
}

fn map_proc_macro_stage(stage: StageName, is_proc_macro_invocation: bool) -> StageName {
    if is_proc_macro_invocation
        && matches!(
            stage,
            StageName::Metadata | StageName::Mir | StageName::LlvmIr | StageName::Obj
        )
    {
        StageName::ProcMacro
    } else {
        stage
    }
}

#[derive(Debug)]
struct StageSideResult {
    status: i32,
    path: Option<PathBuf>,
    sha256: Option<String>,
    stderr_excerpt: String,
}

fn run_single_stage(
    inv: &InvocationRecord,
    side: &str,
    emit: &str,
    ext: &str,
    build_script_env: Option<&[(String, String)]>,
    temp_root: &Path,
) -> Result<StageSideResult> {
    let Some(bin) = inv.argv.first() else {
        anyhow::bail!("invocation had empty argv");
    };

    let mut args = sanitize_args(&inv.argv[1..]);
    let out_path = temp_root.join(format!(
        "{}-{}-{}.{}",
        side,
        inv.crate_name.as_deref().unwrap_or("crate"),
        emit,
        ext
    ));
    if !contains_save_temps(&args) {
        args.push("-C".to_string());
        args.push("save-temps".to_string());
    }
    args.push(format!("--emit={emit}={}", out_path.display()));

    let mut cmd = Command::new(bin);
    cmd.args(&args);
    cmd.current_dir(&inv.cwd);
    for (k, v) in &inv.env {
        cmd.env(k, v);
    }
    if let Some(extra_env) = build_script_env {
        for (k, v) in extra_env {
            cmd.env(k, v);
        }
    }
    let out = cmd
        .output()
        .with_context(|| format!("failed to run stage replay for emit={emit}, tool={bin}"))?;

    let status = out.status.code().unwrap_or(1);
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    let stderr_excerpt = stderr.lines().take(4).collect::<Vec<_>>().join(" | ");

    if status != 0 || !out_path.is_file() {
        return Ok(StageSideResult { status, path: Some(out_path), sha256: None, stderr_excerpt });
    }

    Ok(StageSideResult {
        status,
        path: Some(out_path.clone()),
        sha256: Some(sha256_file(&out_path)?),
        stderr_excerpt,
    })
}

fn sanitize_args(args: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < args.len() {
        let cur = &args[i];

        if cur == "--emit"
            || cur == "--out-dir"
            || cur == "-o"
            || (cur == "-C" && args.get(i + 1).is_some_and(|v| v.starts_with("incremental=")))
        {
            i += 1;
            if i < args.len() {
                i += 1;
            }
            continue;
        }
        if cur.starts_with("--emit=")
            || cur.starts_with("--out-dir=")
            || cur.starts_with("-o")
            || cur.starts_with("-Cincremental=")
        {
            i += 1;
            continue;
        }

        out.push(cur.clone());
        i += 1;
    }
    out
}

fn contains_save_temps(args: &[String]) -> bool {
    let mut i = 0usize;
    while i < args.len() {
        if args[i] == "-C" {
            if let Some(next) = args.get(i + 1) {
                if next == "save-temps" || next.starts_with("save-temps=") {
                    return true;
                }
            }
            i += 1;
        } else if args[i] == "-Csave-temps" || args[i].starts_with("-Csave-temps=") {
            return true;
        }
        i += 1;
    }
    false
}

fn same_multiset(left: &[String], right: &[String]) -> bool {
    if left.len() != right.len() {
        return false;
    }
    let mut counts = std::collections::BTreeMap::<&str, i32>::new();
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

#[cfg(test)]
#[path = "tests/stage.rs"]
mod tests;
