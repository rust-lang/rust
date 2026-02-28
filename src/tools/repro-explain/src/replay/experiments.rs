use std::collections::BTreeMap;
use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result};
use camino::Utf8PathBuf;

use crate::capture::{
    CaptureOptions, capture, load_artifacts, load_build_script_messages, load_build_script_stdout,
};
use crate::cli::{CaptureFlags, ConfirmLevel};
use crate::diff::run_diff;
use crate::model::{
    BuildScriptExecutedMessage, BuildScriptStdoutRecord, CommandKind, DiffClass, DiffStatus,
    ReplayOutcome,
};
use crate::util::process::unix_ts;

#[derive(Debug, Clone)]
pub struct ReplayContext {
    pub work_dir: Utf8PathBuf,
    pub command: Vec<String>,
    pub command_kind: CommandKind,
    pub capture_flags: CaptureFlags,
    pub jobs: usize,
    pub same_source_replay: bool,
    pub focus_package_id: Option<String>,
}

pub fn run_confirmation(
    level: ConfirmLevel,
    class: &DiffClass,
    ctx: &ReplayContext,
) -> Result<Option<ReplayOutcome>> {
    if matches!(level, ConfirmLevel::None) {
        return Ok(None);
    }

    match class {
        DiffClass::Timestamp => Ok(Some(source_date_epoch_replay(ctx)?)),
        DiffClass::ScheduleSensitiveParallelism | DiffClass::UnstableOrder => {
            Ok(Some(serial_jobs_replay(ctx)?))
        }
        DiffClass::BuildScript if matches!(level, ConfirmLevel::Full) => {
            Ok(Some(build_script_replay(ctx)?))
        }
        DiffClass::PathLeak if matches!(level, ConfirmLevel::Full) && ctx.same_source_replay => {
            Ok(Some(same_source_dir_replay(ctx)?))
        }
        _ => Ok(None),
    }
}

fn source_date_epoch_replay(ctx: &ReplayContext) -> Result<ReplayOutcome> {
    let tag = format!("sde-{}", unix_ts());
    let mut extra_env = BTreeMap::new();
    extra_env.insert("SOURCE_DATE_EPOCH".to_string(), "1700000000".to_string());
    let replay_jobs = ctx.jobs.max(1);
    let replay_command = if ctx.command_kind == CommandKind::Cargo {
        with_jobs(&ctx.command, replay_jobs)
    } else {
        ctx.command.clone()
    };

    let run_a = format!("replay-{tag}-A");
    let run_b = format!("replay-{tag}-B");

    let mut flags = ctx.capture_flags.clone();
    flags.keep_target = false;

    let a = capture(CaptureOptions {
        run_id: run_a.clone(),
        work_dir: ctx.work_dir.clone(),
        command: replay_command.clone(),
        flags: flags.clone(),
        extra_env: extra_env.clone(),
        cwd_override: None,
        target_dir_override: None,
    })?;
    let b = capture(CaptureOptions {
        run_id: run_b.clone(),
        work_dir: ctx.work_dir.clone(),
        command: replay_command,
        flags,
        extra_env,
        cwd_override: None,
        target_dir_override: None,
    })?;

    let diff = run_diff(
        &ctx.work_dir,
        &ctx.work_dir.join("runs").join(run_a),
        &ctx.work_dir.join("runs").join(run_b),
    )?;
    let changed = count_changed(&diff.manifest.entries);

    Ok(ReplayOutcome {
        experiment: "source-date-epoch".to_string(),
        success: changed == 0 && a.status == 0 && b.status == 0,
        detail: format!(
            "replayed two runs with SOURCE_DATE_EPOCH=1700000000 and jobs={replay_jobs}, changed_artifacts={changed}, status=({}, {})",
            a.status, b.status
        ),
    })
}

fn serial_jobs_replay(ctx: &ReplayContext) -> Result<ReplayOutcome> {
    if ctx.command_kind != CommandKind::Cargo {
        return Ok(ReplayOutcome {
            experiment: "serial-jobs".to_string(),
            success: false,
            detail: "serial replay currently supports cargo commands only".to_string(),
        });
    }

    let tag = format!("serial-{}", unix_ts());
    let run_a = format!("replay-{tag}-A");
    let run_b = format!("replay-{tag}-B");

    let requested_jobs = ctx.jobs;
    let command = with_jobs_one(&ctx.command);

    let mut flags = ctx.capture_flags.clone();
    flags.keep_target = false;

    let a = capture(CaptureOptions {
        run_id: run_a.clone(),
        work_dir: ctx.work_dir.clone(),
        command: command.clone(),
        flags: flags.clone(),
        extra_env: BTreeMap::new(),
        cwd_override: None,
        target_dir_override: None,
    })?;
    let b = capture(CaptureOptions {
        run_id: run_b.clone(),
        work_dir: ctx.work_dir.clone(),
        command,
        flags,
        extra_env: BTreeMap::new(),
        cwd_override: None,
        target_dir_override: None,
    })?;

    let diff = run_diff(
        &ctx.work_dir,
        &ctx.work_dir.join("runs").join(run_a),
        &ctx.work_dir.join("runs").join(run_b),
    )?;
    let changed = count_changed(&diff.manifest.entries);

    Ok(ReplayOutcome {
        experiment: "serial-jobs".to_string(),
        success: changed == 0 && a.status == 0 && b.status == 0,
        detail: format!(
            "replayed with --jobs 1 (requested jobs={requested_jobs}), changed_artifacts={changed}, status=({}, {})",
            a.status, b.status
        ),
    })
}

fn build_script_replay(ctx: &ReplayContext) -> Result<ReplayOutcome> {
    if ctx.command_kind != CommandKind::Cargo {
        return Ok(ReplayOutcome {
            experiment: "build-script-replay".to_string(),
            success: false,
            detail: "build-script replay currently supports cargo commands only".to_string(),
        });
    }

    let tag = format!("buildrs-{}", unix_ts());
    let run_a = format!("replay-{tag}-A");
    let run_b = format!("replay-{tag}-B");

    let mut flags = ctx.capture_flags.clone();
    flags.keep_target = false;

    let focused_command = if let Some(pkg) = ctx.focus_package_id.as_deref() {
        with_focused_package(&ctx.command, pkg)
    } else {
        ctx.command.clone()
    };
    let replay_jobs = ctx.jobs.max(1);
    let focused_command = with_jobs(&focused_command, replay_jobs);

    let a = capture(CaptureOptions {
        run_id: run_a.clone(),
        work_dir: ctx.work_dir.clone(),
        command: focused_command.clone(),
        flags: flags.clone(),
        extra_env: BTreeMap::new(),
        cwd_override: None,
        target_dir_override: None,
    })?;
    let b = capture(CaptureOptions {
        run_id: run_b.clone(),
        work_dir: ctx.work_dir.clone(),
        command: focused_command,
        flags,
        extra_env: BTreeMap::new(),
        cwd_override: None,
        target_dir_override: None,
    })?;

    let run_a_dir = ctx.work_dir.join("runs").join(&run_a);
    let run_b_dir = ctx.work_dir.join("runs").join(&run_b);

    let left_payload = load_build_script_messages(&run_a_dir)?;
    let right_payload = load_build_script_messages(&run_b_dir)?;
    let payload_changed = left_payload != right_payload;

    let left_out_dir = load_artifacts(&run_a_dir)?
        .into_iter()
        .filter(|a| a.kind == "out-dir-file")
        .map(|a| (a.rel_path, a.sha256))
        .collect::<std::collections::BTreeMap<_, _>>();
    let right_out_dir = load_artifacts(&run_b_dir)?
        .into_iter()
        .filter(|a| a.kind == "out-dir-file")
        .map(|a| (a.rel_path, a.sha256))
        .collect::<std::collections::BTreeMap<_, _>>();
    let out_dir_changed = left_out_dir != right_out_dir;

    let left_stdout = load_build_script_stdout(&run_a_dir)?;
    let right_stdout = load_build_script_stdout(&run_b_dir)?;
    let (instruction_order_changed, instruction_order_only) =
        build_script_instruction_diff(&left_stdout, &right_stdout, ctx.focus_package_id.as_deref());

    let order_only = payload_order_only(&left_payload, &right_payload);

    Ok(ReplayOutcome {
        experiment: "build-script-replay".to_string(),
        success: (payload_changed || out_dir_changed || instruction_order_changed)
            && a.status == 0
            && b.status == 0,
        detail: format!(
            "payload_changed={payload_changed}, out_dir_changed={out_dir_changed}, instruction_order_changed={instruction_order_changed}, instruction_order_only={instruction_order_only}, order_only={order_only}, jobs={replay_jobs}, package_filter={}, status=({}, {})",
            ctx.focus_package_id.as_deref().unwrap_or("<none>"),
            a.status,
            b.status
        ),
    })
}

fn with_jobs(command: &[String], jobs: usize) -> Vec<String> {
    let jobs = jobs.max(1);
    let jobs_value = jobs.to_string();
    let mut out = Vec::with_capacity(command.len() + 2);
    let mut i = 0usize;
    let mut replaced = false;

    while i < command.len() {
        let cur = &command[i];
        if cur == "--jobs" || cur == "-j" {
            out.push(cur.clone());
            out.push(jobs_value.clone());
            replaced = true;
            i += 1;
            if i < command.len() {
                i += 1;
            }
            continue;
        }
        if cur.starts_with("--jobs=") {
            out.push(format!("--jobs={jobs}"));
            replaced = true;
            i += 1;
            continue;
        }
        if cur.starts_with("-j") && cur.len() > 2 {
            out.push(format!("-j{jobs}"));
            replaced = true;
            i += 1;
            continue;
        }

        out.push(cur.clone());
        i += 1;
    }

    if !replaced {
        out.push("--jobs".to_string());
        out.push(jobs_value);
    }
    out
}

fn with_jobs_one(command: &[String]) -> Vec<String> {
    with_jobs(command, 1)
}

fn with_focused_package(command: &[String], package_id: &str) -> Vec<String> {
    if command.is_empty() || command[0] != "cargo" {
        return command.to_vec();
    }
    let mut out = command.to_vec();
    if out.iter().any(|arg| arg == "-p" || arg == "--package" || arg.starts_with("--package=")) {
        return out;
    }
    out.push("--package".to_string());
    out.push(package_id.to_string());
    out
}

fn same_source_dir_replay(ctx: &ReplayContext) -> Result<ReplayOutcome> {
    let tag = format!("same-src-{}", unix_ts());
    let run_a = format!("replay-{tag}-A");
    let run_b = format!("replay-{tag}-B");

    let replay_root = ctx.work_dir.join("replay").join(&tag);
    std::fs::create_dir_all(&replay_root)
        .with_context(|| format!("failed to create replay root {replay_root}"))?;
    let workspace_copy = replay_root.join("workspace");
    let stable_target = replay_root.join("target");

    prepare_workspace_copy(&workspace_copy)?;

    let mut flags = ctx.capture_flags.clone();
    flags.keep_target = false;

    let replay_jobs = ctx.jobs.max(1);
    let replay_command = if ctx.command_kind == CommandKind::Cargo {
        with_jobs(&ctx.command, replay_jobs)
    } else {
        ctx.command.clone()
    };

    let a = capture(CaptureOptions {
        run_id: run_a.clone(),
        work_dir: ctx.work_dir.clone(),
        command: replay_command.clone(),
        flags: flags.clone(),
        extra_env: BTreeMap::new(),
        cwd_override: Some(workspace_copy.clone()),
        target_dir_override: Some(stable_target.clone()),
    })?;
    let b = capture(CaptureOptions {
        run_id: run_b.clone(),
        work_dir: ctx.work_dir.clone(),
        command: replay_command,
        flags,
        extra_env: BTreeMap::new(),
        cwd_override: Some(workspace_copy.clone()),
        target_dir_override: Some(stable_target),
    })?;

    let diff = run_diff(
        &ctx.work_dir,
        &ctx.work_dir.join("runs").join(run_a),
        &ctx.work_dir.join("runs").join(run_b),
    )?;
    let changed = count_changed(&diff.manifest.entries);
    Ok(ReplayOutcome {
        experiment: "same-source-dir-replay".to_string(),
        success: changed == 0 && a.status == 0 && b.status == 0,
        detail: format!(
            "replayed in fixed workspace copy={}, jobs={replay_jobs}, changed_artifacts={changed}, status=({}, {})",
            workspace_copy, a.status, b.status
        ),
    })
}

fn prepare_workspace_copy(dest: &Utf8PathBuf) -> Result<()> {
    if dest.exists() {
        std::fs::remove_dir_all(dest)
            .with_context(|| format!("failed to clean old workspace copy at {dest}"))?;
    }
    std::fs::create_dir_all(dest)
        .with_context(|| format!("failed to create workspace copy dir {dest}"))?;

    let src = std::env::current_dir().context("failed to read current working directory")?;
    let src_utf8 = Utf8PathBuf::from_path_buf(src.clone())
        .map_err(|_| anyhow::anyhow!("workspace path is not utf8: {}", src.display()))?;

    let git_dir = src.join(".git");
    if git_dir.exists() {
        // Prefer worktree for large repositories to avoid a full copy.
        let status = Command::new("git")
            .args(["worktree", "add", "--detach", dest.as_str(), "HEAD"])
            .current_dir(&src)
            .status();
        if let Ok(status) = status {
            if status.success() {
                return Ok(());
            }
        }
        let _ = std::fs::remove_dir_all(dest);
        std::fs::create_dir_all(dest)
            .with_context(|| format!("failed to recreate workspace copy dir {dest}"))?;
    }

    copy_dir_recursively(&src_utf8, dest, dest)?;
    Ok(())
}

fn copy_dir_recursively(
    src_root: &Utf8PathBuf,
    src: &Utf8PathBuf,
    dest: &Utf8PathBuf,
) -> Result<()> {
    for entry in std::fs::read_dir(src).with_context(|| format!("failed to read dir {src}"))? {
        let entry = entry.with_context(|| format!("failed to read dir entry under {src}"))?;
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if name == ".git" || name == "target" || name == ".repro" {
            continue;
        }

        let rel = path
            .strip_prefix(src_root)
            .with_context(|| format!("failed to relativize {} to {}", path.display(), src_root))?;
        let rel_utf8 = Utf8PathBuf::from_path_buf(rel.to_path_buf())
            .map_err(|_| anyhow::anyhow!("non-utf8 path under workspace: {}", path.display()))?;
        let dst = dest_root_join(dest, &rel_utf8);

        if path.is_dir() {
            std::fs::create_dir_all(&dst).with_context(|| format!("failed to create {dst}"))?;
            let path_utf8 = Utf8PathBuf::from_path_buf(path)
                .map_err(|_| anyhow::anyhow!("non-utf8 path under workspace"))?;
            copy_dir_recursively(src_root, &path_utf8, dest)?;
        } else if path.is_file() {
            if let Some(parent) = Path::new(dst.as_str()).parent() {
                std::fs::create_dir_all(parent)
                    .with_context(|| format!("failed to create {}", parent.display()))?;
            }
            std::fs::copy(&path, &dst)
                .with_context(|| format!("failed to copy {} -> {}", path.display(), dst))?;
        }
    }
    Ok(())
}

fn dest_root_join(dest_root: &Utf8PathBuf, rel: &Utf8PathBuf) -> Utf8PathBuf {
    let mut out = dest_root.clone();
    out.push(rel);
    out
}

fn payload_order_only(
    left: &[BuildScriptExecutedMessage],
    right: &[BuildScriptExecutedMessage],
) -> bool {
    if left.len() != right.len() {
        return false;
    }
    if left == right {
        return false;
    }

    left.iter().zip(right.iter()).all(|(l, r)| {
        l.package_id == r.package_id
            && same_multiset(&l.linked_libs, &r.linked_libs)
            && same_multiset(&l.linked_paths, &r.linked_paths)
            && same_multiset(&l.cfgs, &r.cfgs)
            && same_multiset(
                &l.env.iter().map(|(k, v)| format!("{k}={v}")).collect::<Vec<_>>(),
                &r.env.iter().map(|(k, v)| format!("{k}={v}")).collect::<Vec<_>>(),
            )
    })
}

fn build_script_instruction_diff(
    left: &[BuildScriptStdoutRecord],
    right: &[BuildScriptStdoutRecord],
    focus_package_id: Option<&str>,
) -> (bool, bool) {
    let left_map =
        left.iter().map(|r| (r.package_id.clone(), r.lines.clone())).collect::<BTreeMap<_, _>>();
    let right_map =
        right.iter().map(|r| (r.package_id.clone(), r.lines.clone())).collect::<BTreeMap<_, _>>();

    let package_ids = if let Some(pkg) = focus_package_id {
        vec![pkg.to_string()]
    } else {
        left_map
            .keys()
            .chain(right_map.keys())
            .cloned()
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>()
    };

    let mut changed = false;
    let mut all_order_only = true;
    let mut compared_pairs = 0usize;

    for pkg in package_ids {
        let left_lines = left_map.get(&pkg);
        let right_lines = right_map.get(&pkg);
        if left_lines == right_lines {
            continue;
        }
        changed = true;
        let (Some(left_lines), Some(right_lines)) = (left_lines, right_lines) else {
            all_order_only = false;
            continue;
        };
        if left_lines.is_empty() || right_lines.is_empty() {
            all_order_only = false;
            continue;
        }
        compared_pairs += 1;
        if !same_multiset(left_lines, right_lines) {
            all_order_only = false;
        }
    }

    (changed, changed && compared_pairs > 0 && all_order_only)
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

fn count_changed(entries: &[crate::model::DiffEntry]) -> usize {
    entries
        .iter()
        .filter(|e| {
            e.status == DiffStatus::Changed
                || e.status == DiffStatus::LeftOnly
                || e.status == DiffStatus::RightOnly
        })
        .count()
}

#[cfg(test)]
#[path = "tests/experiments.rs"]
mod tests;
