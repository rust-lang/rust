use std::fs;
use std::path::Path;

use anyhow::Result;
use log::info;

pub fn write_bootstrap_toml(env_root: &Path, target: Option<&str>) -> Result<()> {
    let target_line = target.map(|t| format!("target = [\"{}\"]\n", t)).unwrap_or_default();

    let content = format!(
        r#"
[llvm]
download-ci-llvm = true

[build]
extended = true
tools = ["cargo", "clippy", "rustfmt", "rustdoc"]
full-bootstrap = true
{}

[rust]
remap-debuginfo = true
debug = false
debug-assertions = false
backtrace-on-ice = false
debug-logging = false
channel = "nightly"

[dist]
src-tarball = false
"#,
        target_line
    );

    let toml_path = env_root.join("bootstrap.toml");
    fs::write(&toml_path, content.trim_start())?;
    info!("Wrote deterministic bootstrap.toml to {}", toml_path.display());
    Ok(())
}
