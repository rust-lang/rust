use std::fmt::{self, Write};

#[derive(Debug, Clone, Default)]
pub struct ClippyConfiguration {
    pub name: String,
    pub default: String,
    pub lints: Vec<String>,
    pub doc: String,
    pub deprecation_reason: Option<&'static str>,
}

impl fmt::Display for ClippyConfiguration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "- `{}`: {}", self.name, self.doc)?;
        if !self.default.is_empty() {
            write!(f, " (default: `{}`)", self.default)?;
        }
        Ok(())
    }
}

impl ClippyConfiguration {
    pub fn new(
        name: &'static str,
        default: String,
        doc_comment: &'static str,
        deprecation_reason: Option<&'static str>,
    ) -> Self {
        let (lints, doc) = parse_config_field_doc(doc_comment)
            .unwrap_or_else(|| (vec![], "[ERROR] MALFORMED DOC COMMENT".to_string()));

        Self {
            name: to_kebab(name),
            lints,
            doc,
            default,
            deprecation_reason,
        }
    }

    pub fn to_markdown_paragraph(&self) -> String {
        let mut out = format!(
            "## `{}`\n{}\n\n",
            self.name,
            self.doc
                .lines()
                .map(|line| line.strip_prefix("    ").unwrap_or(line))
                .collect::<Vec<_>>()
                .join("\n"),
        );

        if !self.default.is_empty() {
            write!(out, "**Default Value:** `{}`\n\n", self.default).unwrap();
        }

        write!(
            out,
            "---\n**Affected lints:**\n{}\n\n",
            self.lints
                .iter()
                .map(|name| name.to_string().split_whitespace().next().unwrap().to_string())
                .map(|name| format!("* [`{name}`](https://rust-lang.github.io/rust-clippy/master/index.html#{name})"))
                .collect::<Vec<_>>()
                .join("\n"),
        )
        .unwrap();

        out
    }

    pub fn to_markdown_link(&self) -> String {
        const BOOK_CONFIGS_PATH: &str = "https://doc.rust-lang.org/clippy/lint_configuration.html";
        format!("[`{}`]: {BOOK_CONFIGS_PATH}#{}", self.name, self.name)
    }
}

/// This parses the field documentation of the config struct.
///
/// ```rust, ignore
/// parse_config_field_doc(cx, "Lint: LINT_NAME_1, LINT_NAME_2. Papa penguin, papa penguin")
/// ```
///
/// Would yield:
/// ```rust, ignore
/// Some(["lint_name_1", "lint_name_2"], "Papa penguin, papa penguin")
/// ```
fn parse_config_field_doc(doc_comment: &str) -> Option<(Vec<String>, String)> {
    const DOC_START: &str = " Lint: ";
    if doc_comment.starts_with(DOC_START)
        && let Some(split_pos) = doc_comment.find('.')
    {
        let mut doc_comment = doc_comment.to_string();
        let mut documentation = doc_comment.split_off(split_pos);

        // Extract lints
        doc_comment.make_ascii_lowercase();
        let lints: Vec<String> = doc_comment
            .split_off(DOC_START.len())
            .split(", ")
            .map(str::to_string)
            .collect();

        // Format documentation correctly
        // split off leading `.` from lint name list and indent for correct formatting
        documentation = documentation.trim_start_matches('.').trim().replace("\n ", "\n    ");

        Some((lints, documentation))
    } else {
        None
    }
}

/// Transforms a given `snake_case_string` to a tasty `kebab-case-string`
fn to_kebab(config_name: &str) -> String {
    config_name.replace('_', "-")
}
