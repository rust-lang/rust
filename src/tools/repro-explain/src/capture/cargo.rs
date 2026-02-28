use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use camino::Utf8PathBuf;
use serde::Serialize;
use serde_json::Value;

use super::hash::{parse_dep_info, sha256_file, sha256_files_parallel, walk_files};
use super::wrapper::load_invocations;
use crate::cli::CaptureFlags;
use crate::model::{
    ArtifactRecord, BuildScriptExecutedMessage, BuildScriptStdoutRecord, CommandKind,
    CompilerArtifactMessage, EnvMode, HashRecord, InvocationRecord, RunMeta, SCHEMA_VERSION,
};
use crate::util::path_norm::{artifact_kind, normalize_abs, relativize};
use crate::util::process::{run_command, shell_escape_single, utc_now_rfc3339};

#[derive(Debug, Clone)]
pub struct CaptureOptions {
    pub run_id: String,
    pub work_dir: Utf8PathBuf,
    pub command: Vec<String>,
    pub flags: CaptureFlags,
    pub extra_env: BTreeMap<String, String>,
    pub cwd_override: Option<Utf8PathBuf>,
    pub target_dir_override: Option<Utf8PathBuf>,
}

#[derive(Debug, Clone)]
pub struct CaptureResult {
    pub run_dir: Utf8PathBuf,
    pub status: i32,
    pub artifacts: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct OutDirManifest {
    pub entries: Vec<OutDirManifestEntry>,
}

#[derive(Debug, Clone, Serialize)]
pub struct OutDirManifestEntry {
    pub package_id: String,
    pub out_dir: String,
    pub files: Vec<OutDirFileEntry>,
}

#[derive(Debug, Clone, Serialize)]
pub struct OutDirFileEntry {
    pub rel_path: String,
    pub sha256: String,
}

pub fn capture(opts: CaptureOptions) -> Result<CaptureResult> {
    if opts.command.is_empty() {
        anyhow::bail!("build command cannot be empty");
    }

    let cwd = if let Some(override_cwd) = &opts.cwd_override {
        std::path::PathBuf::from(override_cwd)
    } else {
        std::env::current_dir().context("failed to get current working directory")?
    };
    let work_dir = absolutize_path(&opts.work_dir, &cwd)?;
    let run_dir = work_dir.join("runs").join(&opts.run_id);
    let target_dir = opts
        .target_dir_override
        .as_ref()
        .map(|p| absolutize_path(p, &cwd))
        .transpose()?
        .unwrap_or_else(|| run_dir.join("target"));

    if run_dir.exists() {
        fs::remove_dir_all(&run_dir)
            .with_context(|| format!("failed to clean existing run dir {}", run_dir))?;
    }

    fs::create_dir_all(run_dir.join("timings"))
        .with_context(|| format!("failed to create run dir {}", run_dir))?;

    let command_kind = CommandKind::detect(&opts.command);

    write_json(run_dir.join("command.json"), &opts.command)?;

    if !opts.flags.keep_target {
        let _ = fs::remove_dir_all(&target_dir);
    }
    fs::create_dir_all(&target_dir)
        .with_context(|| format!("failed to create target dir {}", target_dir))?;

    let rustc_log = run_dir.join("invocations-rustc.jsonl");
    let rustdoc_log = run_dir.join("invocations-rustdoc.jsonl");
    let rustc_wrapper = run_dir.join("rustc-wrapper.sh");
    let rustdoc_wrapper = run_dir.join("rustdoc-wrapper.sh");
    create_wrapper_script(&rustc_wrapper, "__wrap-rustc")?;
    create_wrapper_script(&rustdoc_wrapper, "__wrap-rustdoc")?;

    if command_kind == CommandKind::Cargo {
        capture_cargo_metadata(&opts.command[0], &cwd, &run_dir)?;
    }

    let mut env_overrides = opts.extra_env.clone();
    env_overrides.insert(
        "REPRO_EXPLAIN_CAPTURE_ALL_ENV".to_string(),
        if opts.flags.capture_all_env { "1" } else { "0" }.to_string(),
    );
    env_overrides.insert("REPRO_EXPLAIN_RUSTC_LOG".to_string(), rustc_log.to_string());
    env_overrides.insert("REPRO_EXPLAIN_RUSTDOC_LOG".to_string(), rustdoc_log.to_string());
    env_overrides.insert(
        "REPRO_EXPLAIN_REAL_RUSTDOC".to_string(),
        std::env::var("RUSTDOC").unwrap_or_else(|_| "rustdoc".to_string()),
    );
    env_overrides.insert("RUSTC_WRAPPER".to_string(), rustc_wrapper.to_string());
    env_overrides.insert("RUSTDOC".to_string(), rustdoc_wrapper.to_string());

    if command_kind == CommandKind::Cargo {
        env_overrides.insert("CARGO_TARGET_DIR".to_string(), target_dir.to_string());
    }

    if opts.flags.binary_dep_depinfo {
        append_env_flag(&mut env_overrides, "RUSTFLAGS", "-Zbinary-dep-depinfo");
    }

    let mut build_cmd = opts.command.clone();
    if command_kind == CommandKind::Cargo
        && !build_cmd.iter().any(|arg| arg.starts_with("--message-format"))
    {
        build_cmd.push("--message-format=json".to_string());
    }

    let output = run_command(&build_cmd, &cwd, &env_overrides)?;
    fs::write(run_dir.join("stdout.log"), &output.stdout).context("failed to write stdout.log")?;
    fs::write(run_dir.join("stderr.log"), &output.stderr).context("failed to write stderr.log")?;
    if command_kind == CommandKind::Cargo {
        copy_cargo_timings(target_dir.as_std_path(), &run_dir.join("timings"))?;
    }

    let (compiler_msgs, build_script_msgs) = parse_cargo_messages(&output.stdout, &run_dir)?;
    write_json(run_dir.join("build-script-executed.json"), &build_script_msgs)?;
    write_json(run_dir.join("compiler-artifacts.json"), &compiler_msgs)?;
    let build_script_stdout = collect_build_script_stdout_records(&build_script_msgs);
    write_json(run_dir.join("build-script-stdout.json"), &build_script_stdout)?;

    let rustc_invocations = load_invocations(rustc_log.as_std_path())?;
    let _rustdoc_invocations = load_invocations(rustdoc_log.as_std_path())?;

    let (artifacts, hashes, out_dir_manifest) = collect_artifacts(
        &compiler_msgs,
        &build_script_msgs,
        &rustc_invocations,
        target_dir.as_std_path(),
        &cwd,
        command_kind != CommandKind::Cargo || compiler_msgs.is_empty(),
    )?;

    write_json(run_dir.join("artifacts.json"), &artifacts)?;
    write_json(run_dir.join("hashes.json"), &hashes)?;
    write_json(run_dir.join("out-dir-manifest.json"), &out_dir_manifest)?;

    let meta = RunMeta {
        schema_version: SCHEMA_VERSION,
        run_id: opts.run_id,
        command: opts.command,
        workspace_root: normalize_abs(&cwd),
        target_dir: target_dir.to_string(),
        env_mode: if opts.flags.capture_all_env { EnvMode::All } else { EnvMode::Allowlist },
        timestamp_utc: utc_now_rfc3339(),
        command_kind,
    };
    write_json(run_dir.join("meta.json"), &meta)?;

    Ok(CaptureResult { run_dir, status: output.status, artifacts: artifacts.len() })
}

fn capture_cargo_metadata(cargo_bin: &str, cwd: &Path, run_dir: &Utf8PathBuf) -> Result<()> {
    let output = std::process::Command::new(cargo_bin)
        .args(["metadata", "--format-version", "1"])
        .current_dir(cwd)
        .output()
        .with_context(|| format!("failed to run `{cargo_bin} metadata --format-version 1`"))?;

    if output.status.success() {
        fs::write(run_dir.join("cargo-metadata.json"), output.stdout)
            .context("failed to write cargo-metadata.json")?;
    } else {
        fs::write(run_dir.join("cargo-metadata.err"), output.stderr)
            .context("failed to write cargo-metadata.err")?;
    }

    Ok(())
}

fn create_wrapper_script(path: &Utf8PathBuf, mode: &str) -> Result<()> {
    let exe = std::env::current_exe().context("failed to resolve current executable")?;
    let script = format!(
        "#!/usr/bin/env bash\nexec {} {} \"$@\"\n",
        shell_escape_single(&exe.to_string_lossy()),
        mode
    );

    fs::write(path, script).with_context(|| format!("failed to write wrapper script {path}"))?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = fs::metadata(path)
            .with_context(|| format!("failed to stat wrapper script {path}"))?
            .permissions();
        perms.set_mode(0o755);
        fs::set_permissions(path, perms)
            .with_context(|| format!("failed to chmod wrapper script {path}"))?;
    }

    Ok(())
}

fn append_env_flag(overrides: &mut BTreeMap<String, String>, key: &str, flag: &str) {
    let joined = std::env::var(key)
        .ok()
        .filter(|v| !v.is_empty())
        .map_or_else(|| flag.to_string(), |cur| format!("{cur} {flag}"));
    overrides.insert(key.to_string(), joined);
}

fn parse_cargo_messages(
    stdout: &str,
    run_dir: &Utf8PathBuf,
) -> Result<(Vec<CompilerArtifactMessage>, Vec<BuildScriptExecutedMessage>)> {
    let mut compiler = Vec::new();
    let mut build_script = Vec::new();
    let mut raw_json_lines = Vec::new();

    for line in stdout.lines() {
        let trimmed = line.trim_start();
        if !trimmed.starts_with('{') {
            continue;
        }
        let Ok(value) = serde_json::from_str::<Value>(trimmed) else {
            continue;
        };
        raw_json_lines.push(trimmed.to_string());

        let Some(reason) = value.get("reason").and_then(Value::as_str) else {
            continue;
        };
        match reason {
            "compiler-artifact" => {
                if let Ok(msg) = serde_json::from_value::<CompilerArtifactMessage>(value.clone()) {
                    compiler.push(msg);
                }
            }
            "build-script-executed" => {
                if let Ok(msg) = serde_json::from_value::<BuildScriptExecutedMessage>(value.clone())
                {
                    build_script.push(msg);
                }
            }
            _ => {}
        }
    }

    let mut body = String::new();
    for line in raw_json_lines {
        body.push_str(&line);
        body.push('\n');
    }
    fs::write(run_dir.join("cargo-messages.jsonl"), body)
        .context("failed to write cargo-messages.jsonl")?;

    Ok((compiler, build_script))
}

fn collect_artifacts(
    compiler_msgs: &[CompilerArtifactMessage],
    build_script_msgs: &[BuildScriptExecutedMessage],
    rustc_invocations: &[InvocationRecord],
    target_dir: &Path,
    workspace_root: &Path,
    use_wrapper_fallback: bool,
) -> Result<(Vec<ArtifactRecord>, Vec<HashRecord>, OutDirManifest)> {
    let mut artifacts = Vec::new();
    let mut hashes = Vec::new();
    let mut seen_paths = HashSet::new();
    let mut next_id = 1usize;
    let mut out_dir_manifest_entries = Vec::new();
    let inv_by_id =
        rustc_invocations.iter().map(|inv| (inv.id.clone(), inv)).collect::<HashMap<_, _>>();
    let compiler_files = compiler_msgs
        .iter()
        .flat_map(|msg| msg.filenames.iter().chain(msg.executable.iter()))
        .map(PathBuf::from)
        .filter(|path| path.is_file())
        .collect::<Vec<_>>();
    let compiler_hashes = sha256_files_parallel(&compiler_files)?;
    let build_script_file_sets = build_script_msgs
        .iter()
        .filter_map(|msg| {
            let out_dir = PathBuf::from(&msg.out_dir);
            if !out_dir.is_dir() {
                return None;
            }
            Some((msg, out_dir.clone(), walk_files(&out_dir)))
        })
        .collect::<Vec<_>>();
    let mut build_script_hash_inputs = build_script_file_sets
        .iter()
        .flat_map(|(_, _, files)| files.iter().cloned())
        .collect::<Vec<_>>();
    build_script_hash_inputs.sort();
    build_script_hash_inputs.dedup();
    let build_script_hashes = sha256_files_parallel(&build_script_hash_inputs)?;

    for msg in compiler_msgs {
        let files =
            msg.filenames.iter().cloned().chain(msg.executable.iter().cloned()).collect::<Vec<_>>();

        for filename in files {
            let path = PathBuf::from(&filename);
            if !path.is_file() {
                continue;
            }
            let canonical = normalize_abs(&path);
            if !seen_paths.insert(canonical.clone()) {
                continue;
            }

            let id = format!("art-{next_id:06}");
            next_id += 1;

            let sha = compiler_hashes
                .get(&path)
                .cloned()
                .map(Ok)
                .unwrap_or_else(|| sha256_file(&path))?;
            let dep_inputs = discover_dep_inputs(&path, rustc_invocations)?;
            let producer =
                infer_producer_invocation(&path, Some(&msg.target.name), rustc_invocations);
            let producer_fingerprint = producer
                .as_ref()
                .and_then(|id| inv_by_id.get(id))
                .map(|inv| invocation_fingerprint(inv, workspace_root));
            let rel_path = relativize(&path, target_dir);
            let kind = artifact_kind(&path);

            artifacts.push(ArtifactRecord {
                id: id.clone(),
                path: canonical.clone(),
                rel_path,
                kind: kind.clone(),
                producer_invocation: producer,
                producer_fingerprint,
                package_id: Some(msg.package_id.clone()),
                target_name: Some(msg.target.name.clone()),
                target_kind: msg.target.kind.clone(),
                fresh: msg.fresh,
                sha256: sha.clone(),
                inputs: dep_inputs,
            });

            hashes.push(HashRecord { artifact_id: id, path: canonical, sha256: sha });
        }
    }

    for (msg, out_dir, files) in build_script_file_sets {
        let mut out_dir_files = Vec::new();
        for file in files {
            let canonical = normalize_abs(&file);
            let sha = build_script_hashes
                .get(&file)
                .cloned()
                .map(Ok)
                .unwrap_or_else(|| sha256_file(&file))?;
            out_dir_files.push(OutDirFileEntry {
                rel_path: relativize(&file, &out_dir),
                sha256: sha.clone(),
            });

            if !seen_paths.insert(canonical.clone()) {
                continue;
            }

            let id = format!("art-{next_id:06}");
            next_id += 1;
            let rel_path = relativize(&file, workspace_root);

            artifacts.push(ArtifactRecord {
                id: id.clone(),
                path: canonical.clone(),
                rel_path,
                kind: "out-dir-file".to_string(),
                producer_invocation: None,
                producer_fingerprint: None,
                package_id: Some(msg.package_id.clone()),
                target_name: None,
                target_kind: vec!["build-script".to_string()],
                fresh: false,
                sha256: sha.clone(),
                inputs: Vec::new(),
            });
            hashes.push(HashRecord { artifact_id: id, path: canonical, sha256: sha });
        }

        out_dir_files.sort_by(|a, b| a.rel_path.cmp(&b.rel_path));
        out_dir_manifest_entries.push(OutDirManifestEntry {
            package_id: msg.package_id.clone(),
            out_dir: normalize_abs(&out_dir),
            files: out_dir_files,
        });
    }

    if use_wrapper_fallback {
        collect_wrapper_fallback_artifacts(
            &mut artifacts,
            &mut hashes,
            &mut seen_paths,
            &mut next_id,
            rustc_invocations,
            workspace_root,
        )?;
    }

    artifacts.sort_by(|a, b| a.id.cmp(&b.id));
    hashes.sort_by(|a, b| a.artifact_id.cmp(&b.artifact_id));
    out_dir_manifest_entries
        .sort_by(|a, b| a.package_id.cmp(&b.package_id).then_with(|| a.out_dir.cmp(&b.out_dir)));

    Ok((artifacts, hashes, OutDirManifest { entries: out_dir_manifest_entries }))
}

fn collect_build_script_stdout_records(
    build_script_msgs: &[BuildScriptExecutedMessage],
) -> Vec<BuildScriptStdoutRecord> {
    let mut by_package = BTreeMap::<String, Vec<String>>::new();

    for msg in build_script_msgs {
        let out_dir = PathBuf::from(&msg.out_dir);
        let Some(build_dir) = out_dir.parent() else {
            continue;
        };
        let output_path = build_dir.join("output");
        let Ok(stdout) = fs::read_to_string(&output_path) else {
            continue;
        };

        let lines = by_package.entry(msg.package_id.clone()).or_default();
        for line in stdout.lines() {
            if let Some(normalized) = normalize_cargo_instruction(line) {
                lines.push(normalized);
            }
        }
    }

    by_package
        .into_iter()
        .map(|(package_id, lines)| BuildScriptStdoutRecord { package_id, lines })
        .collect()
}

fn normalize_cargo_instruction(line: &str) -> Option<String> {
    let trimmed = line.trim();
    if let Some(rest) = trimmed.strip_prefix("cargo::") {
        return Some(format!("cargo::{rest}"));
    }
    trimmed.strip_prefix("cargo:").map(|rest| format!("cargo::{rest}"))
}

fn copy_cargo_timings(target_dir: &Path, timings_dir: &Utf8PathBuf) -> Result<()> {
    let src_root = target_dir.join("cargo-timings");
    if !src_root.is_dir() {
        return Ok(());
    }

    for src in walk_files(&src_root) {
        let Ok(rel) = src.strip_prefix(&src_root) else {
            continue;
        };
        let dst = timings_dir.as_std_path().join(rel);
        if let Some(parent) = dst.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create timings dir {}", parent.display()))?;
        }
        fs::copy(&src, &dst).with_context(|| {
            format!("failed to copy cargo timing file {} -> {}", src.display(), dst.display())
        })?;
    }

    Ok(())
}

fn collect_wrapper_fallback_artifacts(
    artifacts: &mut Vec<ArtifactRecord>,
    hashes: &mut Vec<HashRecord>,
    seen_paths: &mut HashSet<String>,
    next_id: &mut usize,
    rustc_invocations: &[InvocationRecord],
    workspace_root: &Path,
) -> Result<()> {
    let mut out_dirs = BTreeMap::<String, PathBuf>::new();
    let inv_by_id =
        rustc_invocations.iter().map(|inv| (inv.id.clone(), inv)).collect::<HashMap<_, _>>();

    for inv in rustc_invocations {
        let Some(out_dir) = &inv.out_dir else {
            continue;
        };
        let out_path = PathBuf::from(out_dir);
        if !out_path.is_dir() {
            continue;
        }
        out_dirs.entry(normalize_abs(&out_path)).or_insert(out_path);
    }
    let mut fallback_files =
        out_dirs.values().flat_map(|out_dir| walk_files(out_dir)).collect::<Vec<_>>();
    fallback_files.sort();
    fallback_files.dedup();
    let fallback_hash_cache = sha256_files_parallel(&fallback_files)?;

    for out_dir in out_dirs.values() {
        for file in walk_files(out_dir) {
            let canonical = normalize_abs(&file);
            if !seen_paths.insert(canonical.clone()) {
                continue;
            }

            let producer = infer_producer_invocation(&file, None, rustc_invocations);
            let producer_fingerprint = producer
                .as_ref()
                .and_then(|id| inv_by_id.get(id))
                .map(|inv| invocation_fingerprint(inv, workspace_root));
            let target_name = producer
                .as_ref()
                .and_then(|id| inv_by_id.get(id))
                .and_then(|inv| inv.crate_name.clone());
            let target_kind = producer
                .as_ref()
                .and_then(|id| inv_by_id.get(id))
                .map(|inv| inv.crate_types.clone())
                .unwrap_or_default();
            let package_id = producer
                .as_ref()
                .and_then(|id| inv_by_id.get(id))
                .and_then(|inv| inv.package_id.clone());

            let id = format!("art-{next_id:06}");
            *next_id += 1;
            let sha = fallback_hash_cache
                .get(&file)
                .cloned()
                .map(Ok)
                .unwrap_or_else(|| sha256_file(&file))?;
            let rel_path = relativize(&file, workspace_root);
            let kind = artifact_kind(&file);
            let inputs = discover_dep_inputs(&file, rustc_invocations)?;

            artifacts.push(ArtifactRecord {
                id: id.clone(),
                path: canonical.clone(),
                rel_path,
                kind,
                producer_invocation: producer,
                producer_fingerprint,
                package_id,
                target_name,
                target_kind,
                fresh: false,
                sha256: sha.clone(),
                inputs,
            });
            hashes.push(HashRecord { artifact_id: id, path: canonical, sha256: sha });
        }
    }

    Ok(())
}

fn discover_dep_inputs(path: &Path, invocations: &[InvocationRecord]) -> Result<Vec<String>> {
    let mut candidates = Vec::new();

    if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
        if let Some(parent) = path.parent() {
            candidates.push(parent.join(format!("{stem}.d")));
        }
    }

    for inv in invocations {
        if let Some(dep) = &inv.dep_info {
            candidates.push(PathBuf::from(dep));
        }
    }

    for candidate in candidates {
        if candidate.is_file() {
            let deps = parse_dep_info(&candidate)?;
            let normalized =
                deps.into_iter().map(|d| normalize_abs(Path::new(&d))).collect::<Vec<_>>();
            return Ok(normalized);
        }
    }

    Ok(Vec::new())
}

fn infer_producer_invocation(
    artifact_path: &Path,
    target_name: Option<&str>,
    invocations: &[InvocationRecord],
) -> Option<String> {
    let mut matches = Vec::new();

    for inv in invocations {
        if let Some(target_name) = target_name {
            if inv.crate_name.as_deref() != Some(target_name) {
                continue;
            }
        }

        if let Some(out_dir) = &inv.out_dir {
            let out = Path::new(out_dir);
            if artifact_path.starts_with(out) {
                matches.push((out.components().count(), inv.id.clone()));
            }
        }
    }

    matches.into_iter().max_by_key(|(depth, _)| *depth).map(|(_, id)| id)
}

fn invocation_fingerprint(inv: &InvocationRecord, workspace_root: &Path) -> String {
    let crate_types =
        if inv.crate_types.is_empty() { "<none>".to_string() } else { inv.crate_types.join(",") };
    let src = inv
        .src_path
        .as_ref()
        .map(|s| {
            let p = Path::new(s);
            if p.is_absolute() { relativize(p, workspace_root) } else { s.clone() }
        })
        .unwrap_or_else(|| "<none>".to_string());
    format!(
        "tool={};crate_name={};crate_types={};src={};target={};debuginfo={}",
        inv.tool,
        inv.crate_name.as_deref().unwrap_or("<none>"),
        crate_types,
        src,
        inv.target_triple.as_deref().unwrap_or("<none>"),
        inv.profile_debuginfo.as_deref().unwrap_or("<none>")
    )
}

fn absolutize_path(path: &Utf8PathBuf, base: &Path) -> Result<Utf8PathBuf> {
    if path.is_absolute() {
        return Ok(path.clone());
    }
    Utf8PathBuf::from_path_buf(base.join(path))
        .map_err(|_| anyhow::anyhow!("non-utf8 path after absolutize: {}", base.display()))
}

fn write_json(path: Utf8PathBuf, value: &impl serde::Serialize) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(value).context("failed to serialize json")?;
    fs::write(&path, bytes).with_context(|| format!("failed to write {}", path))?;
    Ok(())
}

pub fn load_run_meta(run_dir: &Utf8PathBuf) -> Result<RunMeta> {
    let data = fs::read(run_dir.join("meta.json"))
        .with_context(|| format!("failed to read {}/meta.json", run_dir))?;
    serde_json::from_slice(&data).context("failed to parse meta.json")
}

pub fn load_artifacts(run_dir: &Utf8PathBuf) -> Result<Vec<ArtifactRecord>> {
    let data = fs::read(run_dir.join("artifacts.json"))
        .with_context(|| format!("failed to read {}/artifacts.json", run_dir))?;
    serde_json::from_slice(&data).context("failed to parse artifacts.json")
}

pub fn load_invocation_sets(
    run_dir: &Utf8PathBuf,
) -> Result<HashMap<String, Vec<InvocationRecord>>> {
    let rustc = load_invocations(run_dir.join("invocations-rustc.jsonl").as_std_path())?;
    let rustdoc = load_invocations(run_dir.join("invocations-rustdoc.jsonl").as_std_path())?;
    let mut map = HashMap::new();
    map.insert("rustc".to_string(), rustc);
    map.insert("rustdoc".to_string(), rustdoc);
    Ok(map)
}

pub fn load_build_script_messages(
    run_dir: &Utf8PathBuf,
) -> Result<Vec<BuildScriptExecutedMessage>> {
    let data = fs::read(run_dir.join("build-script-executed.json"))
        .with_context(|| format!("failed to read {}/build-script-executed.json", run_dir))?;
    serde_json::from_slice(&data).context("failed to parse build-script-executed.json")
}

pub fn load_build_script_stdout(run_dir: &Utf8PathBuf) -> Result<Vec<BuildScriptStdoutRecord>> {
    let Ok(data) = fs::read(run_dir.join("build-script-stdout.json")) else {
        return Ok(Vec::new());
    };
    serde_json::from_slice(&data).context("failed to parse build-script-stdout.json")
}

pub fn load_compiler_artifact_messages(
    run_dir: &Utf8PathBuf,
) -> Result<Vec<CompilerArtifactMessage>> {
    let data = fs::read(run_dir.join("compiler-artifacts.json"))
        .with_context(|| format!("failed to read {}/compiler-artifacts.json", run_dir))?;
    serde_json::from_slice(&data).context("failed to parse compiler-artifacts.json")
}

#[cfg(test)]
#[path = "tests/cargo.rs"]
mod tests;
