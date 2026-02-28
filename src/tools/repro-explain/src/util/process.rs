use std::collections::BTreeMap;
use std::env;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::{Context, Result};

#[derive(Debug, Clone)]
pub struct CommandOutput {
    pub status: i32,
    pub stdout: String,
    pub stderr: String,
}

pub fn run_command(
    command: &[String],
    cwd: &Path,
    env_overrides: &BTreeMap<String, String>,
) -> Result<CommandOutput> {
    let Some(bin) = command.first() else {
        anyhow::bail!("empty command");
    };

    let mut cmd = Command::new(bin);
    cmd.args(&command[1..]);
    cmd.current_dir(cwd);
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());
    cmd.envs(env_overrides);

    let out = cmd.output().with_context(|| format!("failed to execute command: {command:?}"))?;

    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();

    if !stdout.is_empty() {
        print!("{stdout}");
    }
    if !stderr.is_empty() {
        eprint!("{stderr}");
    }

    Ok(CommandOutput { status: out.status.code().unwrap_or(1), stdout, stderr })
}

pub fn find_on_path(bin: &str) -> Option<PathBuf> {
    let path = env::var_os("PATH")?;
    let paths = env::split_paths(&path);
    for dir in paths {
        let candidate = dir.join(bin);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

pub fn unix_ts() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

pub fn utc_now_rfc3339() -> String {
    if let Ok(out) = Command::new("date").args(["-u", "+%Y-%m-%dT%H:%M:%SZ"]).output() {
        if out.status.success() {
            let formatted = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if !formatted.is_empty() {
                return formatted;
            }
        }
    }
    "1970-01-01T00:00:00Z".to_string()
}

pub fn shell_escape_single(value: &str) -> String {
    if value.is_empty() {
        return "''".to_string();
    }
    format!("'{}'", value.replace('\'', "'\\''"))
}
