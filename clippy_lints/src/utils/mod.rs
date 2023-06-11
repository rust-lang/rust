pub mod author;
pub mod conf;
pub mod dump_hir;
pub mod format_args_collector;
#[cfg(feature = "internal")]
pub mod internal_lints;
#[cfg(feature = "internal")]
use itertools::Itertools;

/// Transforms a given `snake_case_string` to a tasty `kebab-case-string`
fn to_kebab(config_name: &str) -> String {
    config_name.replace('_', "-")
}

#[cfg(feature = "internal")]
const BOOK_CONFIGS_PATH: &str = "https://doc.rust-lang.org/clippy/lint_configuration.html";

// ==================================================================
// Configuration
// ==================================================================
#[derive(Debug, Clone, Default)] //~ ERROR no such field
pub struct ClippyConfiguration {
    pub name: String,
    #[allow(dead_code)]
    config_type: &'static str,
    pub default: String,
    pub lints: Vec<String>,
    pub doc: String,
    #[allow(dead_code)]
    deprecation_reason: Option<&'static str>,
}

impl ClippyConfiguration {
    pub fn new(
        name: &'static str,
        config_type: &'static str,
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
            config_type,
            default,
            deprecation_reason,
        }
    }

    #[cfg(feature = "internal")]
    fn to_markdown_paragraph(&self) -> String {
        format!(
            "## `{}`\n{}\n\n**Default Value:** `{}` (`{}`)\n\n---\n**Affected lints:**\n{}\n\n",
            self.name,
            self.doc
                .lines()
                .map(|line| line.strip_prefix("    ").unwrap_or(line))
                .join("\n"),
            self.default,
            self.config_type,
            self.lints
                .iter()
                .map(|name| name.to_string().split_whitespace().next().unwrap().to_string())
                .map(|name| format!("* [`{name}`](https://rust-lang.github.io/rust-clippy/master/index.html#{name})"))
                .join("\n"),
        )
    }
    #[cfg(feature = "internal")]
    fn to_markdown_link(&self) -> String {
        format!("[`{}`]: {BOOK_CONFIGS_PATH}#{}", self.name, self.name)
    }
}

#[cfg(feature = "internal")]
fn collect_configs() -> Vec<ClippyConfiguration> {
    crate::utils::conf::metadata::get_configuration_metadata()
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
    if_chain! {
        if doc_comment.starts_with(DOC_START);
        if let Some(split_pos) = doc_comment.find('.');
        then {
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
}

// Shamelessly stolen from find_all (https://github.com/nectariner/find_all)
pub trait FindAll: Iterator + Sized {
    fn find_all<P>(&mut self, predicate: P) -> Option<Vec<usize>>
    where
        P: FnMut(&Self::Item) -> bool;
}

impl<I> FindAll for I
where
    I: Iterator,
{
    fn find_all<P>(&mut self, mut predicate: P) -> Option<Vec<usize>>
    where
        P: FnMut(&Self::Item) -> bool,
    {
        let mut occurences = Vec::<usize>::default();
        for (index, element) in self.enumerate() {
            if predicate(&element) {
                occurences.push(index);
            }
        }

        match occurences.len() {
            0 => None,
            _ => Some(occurences),
        }
    }
}
