use std::env;
use std::path::Path;
use std::process::{Command, Stdio};

use anyhow::{Context, Result};
use log::{info, warn};

// Runs x.py in the given environment root. Handles the build or dist command,
// stage limiting, and job config.
pub fn run_xpy(env_root: &Path, jobs: u32, target: Option<&str>, full_dist: bool) -> Result<()> {
    let x_py = env_root.join("x.py");

    let python = env::var("BOOTSTRAP_PYTHON").unwrap_or_else(|_| {
        if cfg!(windows) { "python".to_string() } else { "python3".to_string() }
    });

    let mut cmd = Command::new(&python);
    cmd.arg(&x_py);

    let build_cmd = if full_dist { "dist" } else { "build" };
    cmd.arg(build_cmd);

    if !full_dist {
        cmd.arg("--stage").arg("2");
        cmd.arg("compiler");
    }

    if let Some(t) = target {
        cmd.arg("--target").arg(t);
    }

    cmd.arg("-j").arg(jobs.to_string());
    cmd.arg("--config").arg("bootstrap.toml");
    cmd.current_dir(env_root);
    cmd.stdout(Stdio::inherit());
    cmd.stderr(Stdio::inherit());

    info!("Kicking off: {} {}", python, x_py.display());

    let status = cmd.status().with_context(|| format!("Couldn't run x.py in {:?}", env_root))?;
    if !status.success() {
        return Err(anyhow::anyhow!("Build bombed in {:?}", env_root));
    }

    Ok(())
}

// Figures out the host triple by asking rustc, or guessing if that fails.
pub fn detect_host(src_root: &Path) -> Result<String> {
    let output = Command::new("rustc")
        .arg("-vV")
        .output()
        .context("Couldn't query rustc for version info")?;

    if !output.status.success() {
        warn!("rustc -vV didn't work; falling back to a guess.");
    }

    let out_str = String::from_utf8_lossy(&output.stdout);
    for line in out_str.lines() {
        if line.starts_with("host: ") {
            return Ok(line.trim_start_matches("host: ").trim().to_string());
        }
    }

    let arch = if cfg!(target_arch = "x86_64") {
        "x86_64"
    } else if cfg!(target_arch = "aarch64") {
        "aarch64"
    } else {
        "unknown"
    };
    let os = if cfg!(target_os = "windows") {
        "windows"
    } else if cfg!(target_os = "macos") {
        "apple-darwin"
    } else {
        "linux-gnu"
    };
    info!("Detected host from src root {:?}: {arch}-unknown-{os}", src_root);
    Ok(format!("{arch}-unknown-{os}"))
}
