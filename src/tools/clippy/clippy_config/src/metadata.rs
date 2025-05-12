use itertools::Itertools;
use std::fmt;

#[derive(Debug, Clone, Default)]
pub struct ClippyConfiguration {
    pub name: String,
    pub default: String,
    pub lints: &'static [&'static str],
    pub doc: &'static str,
    pub deprecation_reason: Option<&'static str>,
}

impl fmt::Display for ClippyConfiguration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "- `{}`: {}", self.name, self.doc)?;
        if !self.default.is_empty() {
            write!(f, "\n\n   (default: `{}`)", self.default)?;
        }
        Ok(())
    }
}

impl ClippyConfiguration {
    pub fn to_markdown_paragraph(&self) -> String {
        format!(
            "## `{}`\n{}\n\n**Default Value:** `{}`\n\n---\n**Affected lints:**\n{}\n\n",
            self.name,
            self.doc.lines().map(|x| x.strip_prefix(' ').unwrap_or(x)).join("\n"),
            self.default,
            self.lints.iter().format_with("\n", |name, f| f(&format_args!(
                "* [`{name}`](https://rust-lang.github.io/rust-clippy/master/index.html#{name})"
            ))),
        )
    }

    pub fn to_markdown_link(&self) -> String {
        const BOOK_CONFIGS_PATH: &str = "https://doc.rust-lang.org/clippy/lint_configuration.html";
        format!("[`{}`]: {BOOK_CONFIGS_PATH}#{}", self.name, self.name)
    }
}
