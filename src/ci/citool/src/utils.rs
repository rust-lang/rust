use std::borrow::Cow;
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

/// Outputs a HTML <details> section with the provided summary.
/// Output printed by `func` will be contained within the section.
pub fn output_details<F>(summary: &str, func: F)
where
    F: FnOnce(),
{
    println!(
        r"<details>
<summary>{summary}</summary>
"
    );
    func();
    println!("</details>\n");
}

/// Normalizes Windows-style path delimiters to Unix-style paths.
pub fn normalize_path_delimiters(name: &str) -> Cow<'_, str> {
    if name.contains("\\") { name.replace('\\', "/").into() } else { name.into() }
}
