use crate::Config;

/// Returns `true` if the given span is a part of generated files.
pub(super) fn is_generated_file(original_snippet: &str, config: &Config) -> bool {
    original_snippet
        .lines()
        // looking for marker only in the beginning of the file
        .take(config.generated_marker_line_search_limit())
        .any(|line| line.contains("@generated"))
}
