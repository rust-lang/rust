/// Returns `true` if the given span is a part of generated files.
pub(super) fn is_generated_file(original_snippet: &str) -> bool {
    original_snippet
        .lines()
        .take(5) // looking for marker only in the beginning of the file
        .any(|line| line.contains("@generated"))
}
