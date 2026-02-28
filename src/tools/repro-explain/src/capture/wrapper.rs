use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};

use crate::model::InvocationRecord;
use crate::util::redact::capture_env;

pub fn run_rustc_wrapper(argv: Vec<String>) -> Result<i32> {
    let Some(real_rustc) = argv.first() else {
        anyhow::bail!("rustc wrapper expected real rustc path at argv[0]");
    };

    let tool_args = argv[1..].to_vec();
    let start = unix_ts();
    let status = Command::new(real_rustc)
        .args(&tool_args)
        .status()
        .context("failed to execute wrapped rustc")?;
    let end = unix_ts();

    let rec = make_record("rustc", real_rustc, &tool_args, status.code().unwrap_or(1), start, end);
    append_record_from_env("REPRO_EXPLAIN_RUSTC_LOG", &rec)?;

    Ok(status.code().unwrap_or(1))
}

pub fn run_rustdoc_wrapper(argv: Vec<String>) -> Result<i32> {
    let real_rustdoc = std::env::var("REPRO_EXPLAIN_REAL_RUSTDOC")
        .ok()
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| "rustdoc".to_string());

    let start = unix_ts();
    let status = Command::new(&real_rustdoc)
        .args(&argv)
        .status()
        .context("failed to execute wrapped rustdoc")?;
    let end = unix_ts();

    let rec = make_record("rustdoc", &real_rustdoc, &argv, status.code().unwrap_or(1), start, end);
    append_record_from_env("REPRO_EXPLAIN_RUSTDOC_LOG", &rec)?;

    Ok(status.code().unwrap_or(1))
}

fn make_record(
    tool: &str,
    real_tool: &str,
    tool_args: &[String],
    exit_code: i32,
    start: u64,
    end: u64,
) -> InvocationRecord {
    let capture_all = std::env::var("REPRO_EXPLAIN_CAPTURE_ALL_ENV").ok().as_deref() == Some("1");
    let env = capture_env(capture_all);

    let mut argv = vec![real_tool.to_string()];
    argv.extend_from_slice(tool_args);

    let crate_name = arg_value(tool_args, "--crate-name");
    let crate_types = collect_multi_values(tool_args, "--crate-type");
    let out_dir = arg_value(tool_args, "--out-dir");
    let dep_info = parse_dep_info_arg(tool_args);
    let src_path = find_src_path(tool_args);
    let target_triple = arg_value(tool_args, "--target");
    let profile_debuginfo = extract_debuginfo(tool_args);

    InvocationRecord {
        id: make_invocation_id(),
        tool: tool.to_string(),
        argv,
        cwd: std::env::current_dir()
            .ok()
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_else(|| "<unknown>".to_string()),
        env,
        crate_name,
        crate_types,
        src_path,
        out_dir,
        dep_info,
        package_id: std::env::var("CARGO_PKG_ID").ok(),
        target_triple,
        profile_debuginfo,
        start_timestamp_unix: start,
        end_timestamp_unix: end,
        exit_code,
    }
}

fn append_record_from_env(log_env: &str, rec: &InvocationRecord) -> Result<()> {
    let Some(path) = std::env::var_os(log_env) else {
        return Ok(());
    };

    if let Some(parent) = Path::new(&path).parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create wrapper log dir {}", parent.display()))?;
    }

    let mut f =
        OpenOptions::new().create(true).append(true).open(&path).with_context(|| {
            format!("failed to open wrapper log {}", Path::new(&path).display())
        })?;

    let mut line = serde_json::to_vec(rec).context("failed to serialize invocation record")?;
    line.push(b'\n');
    f.write_all(&line).context("failed to append invocation record to wrapper log")?;
    Ok(())
}

fn make_invocation_id() -> String {
    let ts = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_nanos());
    format!("inv-{ts}-{}", std::process::id())
}

fn unix_ts() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

fn arg_value(args: &[String], flag: &str) -> Option<String> {
    let mut i = 0;
    while i < args.len() {
        let cur = &args[i];
        if cur == flag {
            if let Some(v) = args.get(i + 1) {
                return Some(v.clone());
            }
        } else if let Some(rest) = cur.strip_prefix(&(flag.to_string() + "=")) {
            return Some(rest.to_string());
        }
        i += 1;
    }
    None
}

fn collect_multi_values(args: &[String], flag: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < args.len() {
        let cur = &args[i];
        if cur == flag {
            if let Some(v) = args.get(i + 1) {
                out.extend(v.split(',').map(|s| s.to_string()));
            }
            i += 1;
        } else if let Some(rest) = cur.strip_prefix(&(flag.to_string() + "=")) {
            out.extend(rest.split(',').map(|s| s.to_string()));
        }
        i += 1;
    }
    out
}

fn parse_dep_info_arg(args: &[String]) -> Option<String> {
    for arg in args {
        if let Some(rest) = arg.strip_prefix("--emit=") {
            for item in rest.split(',') {
                if let Some(path) = item.strip_prefix("dep-info=") {
                    return Some(path.to_string());
                }
            }
        }
    }
    None
}

fn find_src_path(args: &[String]) -> Option<String> {
    args.iter().find_map(|arg| {
        if arg.starts_with('-') {
            return None;
        }
        let p = Path::new(arg);
        if p.extension().and_then(|s| s.to_str()) == Some("rs") { Some(arg.clone()) } else { None }
    })
}

fn extract_debuginfo(args: &[String]) -> Option<String> {
    let mut i = 0;
    while i < args.len() {
        if args[i] == "-C" {
            if let Some(next) = args.get(i + 1) {
                if let Some(v) = next.strip_prefix("debuginfo=") {
                    return Some(v.to_string());
                }
            }
            i += 1;
        } else if let Some(v) = args[i].strip_prefix("-Cdebuginfo=") {
            return Some(v.to_string());
        }
        i += 1;
    }
    None
}

pub fn load_invocations(path: &Path) -> Result<Vec<InvocationRecord>> {
    let Ok(data) = std::fs::read_to_string(path) else {
        return Ok(Vec::new());
    };

    data.lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            serde_json::from_str::<InvocationRecord>(line).context("bad invocation json line")
        })
        .collect()
}
