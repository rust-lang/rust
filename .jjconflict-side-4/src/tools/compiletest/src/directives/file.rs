use camino::Utf8Path;

use crate::directives::LineNumber;
use crate::directives::line::{DirectiveLine, line_directive};

pub(crate) struct FileDirectives<'a> {
    pub(crate) path: &'a Utf8Path,
    pub(crate) lines: Vec<DirectiveLine<'a>>,

    /// Whether the test source file contains an explicit `#![no_std]`/`#![no_core]` attribute.
    pub(crate) has_explicit_no_std_core_attribute: bool,
}

impl<'a> FileDirectives<'a> {
    pub(crate) fn from_file_contents(path: &'a Utf8Path, file_contents: &'a str) -> Self {
        let mut lines = vec![];
        let mut has_explicit_no_std_core_attribute = false;

        for (line_number, ln) in LineNumber::enumerate().zip(file_contents.lines()) {
            let ln = ln.trim();

            // Perform a naive check for lines starting with `#![no_std]`/`#![no_core]`, which
            // suppresses the implied `//@ needs-target-std` in codegen tests. This ignores
            // occurrences in ordinary comments.
            //
            // This check is imperfect in some edge cases, but we can generally trust our own test
            // suite to not hit those edge cases (e.g. `#![no_std]`/`#![no_core]` in multi-line
            // comments or string literals). Tests can write `//@ needs-target-std` manually if
            // needed.
            if ln.starts_with("#![no_std]") || ln.starts_with("#![no_core]") {
                has_explicit_no_std_core_attribute = true;
                continue;
            }

            if let Some(directive_line) = line_directive(path, line_number, ln) {
                lines.push(directive_line);
            }
        }

        Self { path, lines, has_explicit_no_std_core_attribute }
    }
}
