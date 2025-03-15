use std::path::Path;

use anyhow::Context;

pub fn load_env_var(name: &str) -> anyhow::Result<String> {
    std::env::var(name).with_context(|| format!("Cannot find environment variable `{name}`"))
}

pub fn read_to_string<P: AsRef<Path>>(path: P) -> anyhow::Result<String> {
    std::fs::read_to_string(&path).with_context(|| format!("Cannot read file {:?}", path.as_ref()))
}

pub fn pluralize(text: &str, count: usize) -> String {
    if count == 1 { text.to_string() } else { format!("{text}s") }
}
