use std::path::Path;

use anyhow::Context;

pub fn load_env_var(name: &str) -> anyhow::Result<String> {
    std::env::var(name).with_context(|| format!("Cannot find environment variable `{name}`"))
}

pub fn read_to_string<P: AsRef<Path>>(path: P) -> anyhow::Result<String> {
    let error = format!("Cannot read file {:?}", path.as_ref());
    std::fs::read_to_string(path).context(error)
}
