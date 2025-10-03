const COMPILETEST_DIRECTIVE_PREFIX: &str = "//@";

/// If the given line begins with the appropriate comment prefix for a directive,
/// returns a struct containing various parts of the directive.
pub(crate) fn line_directive<'line>(
    line_number: usize,
    original_line: &'line str,
) -> Option<DirectiveLine<'line>> {
    // Ignore lines that don't start with the comment prefix.
    let after_comment =
        original_line.trim_start().strip_prefix(COMPILETEST_DIRECTIVE_PREFIX)?.trim_start();

    let revision;
    let raw_directive;

    if let Some(after_open_bracket) = after_comment.strip_prefix('[') {
        // A comment like `//@[foo]` only applies to revision `foo`.
        let Some((line_revision, after_close_bracket)) = after_open_bracket.split_once(']') else {
            panic!(
                "malformed condition directive: expected `{COMPILETEST_DIRECTIVE_PREFIX}[foo]`, found `{original_line}`"
            )
        };

        revision = Some(line_revision);
        raw_directive = after_close_bracket.trim_start();
    } else {
        revision = None;
        raw_directive = after_comment;
    };

    Some(DirectiveLine { line_number, revision, raw_directive })
}

/// The (partly) broken-down contents of a line containing a test directive,
/// which `iter_directives` passes to its callback function.
///
/// For example:
///
/// ```text
/// //@ compile-flags: -O
///     ^^^^^^^^^^^^^^^^^ raw_directive
///
/// //@ [foo] compile-flags: -O
///      ^^^                    revision
///           ^^^^^^^^^^^^^^^^^ raw_directive
/// ```
pub(crate) struct DirectiveLine<'ln> {
    pub(crate) line_number: usize,

    /// Some test directives start with a revision name in square brackets
    /// (e.g. `[foo]`), and only apply to that revision of the test.
    /// If present, this field contains the revision name (e.g. `foo`).
    pub(crate) revision: Option<&'ln str>,

    /// The main part of the directive, after removing the comment prefix
    /// and the optional revision specifier.
    ///
    /// This is "raw" because the directive's name and colon-separated value
    /// (if present) have not yet been extracted or checked.
    pub(crate) raw_directive: &'ln str,
}

impl<'ln> DirectiveLine<'ln> {
    pub(crate) fn applies_to_test_revision(&self, test_revision: Option<&str>) -> bool {
        self.revision.is_none() || self.revision == test_revision
    }
}
