use std::fmt;

use camino::Utf8Path;

const COMPILETEST_DIRECTIVE_PREFIX: &str = "//@";

/// If the given line begins with the appropriate comment prefix for a directive,
/// returns a struct containing various parts of the directive.
pub(crate) fn line_directive<'a>(
    file_path: &'a Utf8Path,
    line_number: usize,
    original_line: &'a str,
) -> Option<DirectiveLine<'a>> {
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

    // The directive name ends at the first occurrence of colon, space, or end-of-string.
    let name = raw_directive.split([':', ' ']).next().expect("split is never empty");

    Some(DirectiveLine { file_path, line_number, revision, raw_directive, name })
}

/// The (partly) broken-down contents of a line containing a test directive,
/// which `iter_directives` passes to its callback function.
///
/// For example:
///
/// ```text
/// //@ compile-flags: -O
///     ^^^^^^^^^^^^^^^^^ raw_directive
///     ^^^^^^^^^^^^^     name
///
/// //@ [foo] compile-flags: -O
///      ^^^                    revision
///           ^^^^^^^^^^^^^^^^^ raw_directive
///           ^^^^^^^^^^^^^     name
/// ```
pub(crate) struct DirectiveLine<'a> {
    /// Path of the file containing this line.
    ///
    /// Mostly used for diagnostics, but some directives (e.g. `//@ pp-exact`)
    /// also use it to compute a value based on the filename.
    pub(crate) file_path: &'a Utf8Path,
    pub(crate) line_number: usize,

    /// Some test directives start with a revision name in square brackets
    /// (e.g. `[foo]`), and only apply to that revision of the test.
    /// If present, this field contains the revision name (e.g. `foo`).
    pub(crate) revision: Option<&'a str>,

    /// The main part of the directive, after removing the comment prefix
    /// and the optional revision specifier.
    ///
    /// This is "raw" because the directive's name and colon-separated value
    /// (if present) have not yet been extracted or checked.
    raw_directive: &'a str,

    /// Name of the directive.
    ///
    /// Invariant: `self.raw_directive.starts_with(self.name)`
    pub(crate) name: &'a str,
}

impl<'ln> DirectiveLine<'ln> {
    pub(crate) fn applies_to_test_revision(&self, test_revision: Option<&str>) -> bool {
        self.revision.is_none() || self.revision == test_revision
    }

    /// Helper method used by `value_after_colon` and `remark_after_space`.
    /// Don't call this directly.
    fn rest_after_separator(&self, separator: u8) -> Option<&'ln str> {
        let n = self.name.len();
        if self.raw_directive.as_bytes().get(n) != Some(&separator) {
            return None;
        }

        Some(&self.raw_directive[n + 1..])
    }

    /// If this directive uses `name: value` syntax, returns the part after
    /// the colon character.
    pub(crate) fn value_after_colon(&self) -> Option<&'ln str> {
        self.rest_after_separator(b':')
    }

    /// If this directive uses `name remark` syntax, returns the part after
    /// the separating space.
    pub(crate) fn remark_after_space(&self) -> Option<&'ln str> {
        self.rest_after_separator(b' ')
    }

    /// Allows callers to print `raw_directive` if necessary,
    /// without accessing the field directly.
    pub(crate) fn display(&self) -> impl fmt::Display {
        self.raw_directive
    }
}
