use camino::Utf8Path;

use crate::directives::LineNumber;
use crate::directives::line::{DirectiveLine, line_directive};

pub(crate) struct FileDirectives<'a> {
    pub(crate) path: &'a Utf8Path,
    pub(crate) lines: Vec<DirectiveLine<'a>>,
}

impl<'a> FileDirectives<'a> {
    /// Create a new [`FileDirectives`] by iterating through the lines of `file_contents`.
    ///
    /// # Note
    ///
    /// When the `suite` argument matches [`crate::common::TestSuite::CodegenLlvm`] a synthetic
    /// `needs-target-std` directive is inserted if needed - that is, if the file does not contain a
    /// `#![no_std]` annotation.
    ///
    /// The objective of this addition is that of making it at least a bit easier to run
    /// the codegen-llvm test suite for targets that do not have a stdlib, without forcing test
    /// writers to remember yet another directive.
    pub(crate) fn from_file_contents(
        suite: crate::common::TestSuite,
        path: &'a Utf8Path,
        file_contents: &'a str,
    ) -> Self {
        let mut lines = vec![];
        let mut generate_needs_std_stub = true;

        for (line_number, ln) in LineNumber::enumerate().zip(file_contents.lines()) {
            let ln = ln.trim();

            if let Some(directive_line) = line_directive(path, line_number, ln) {
                if directive_line.name == "needs-target-std" {
                    generate_needs_std_stub = false;
                }
                lines.push(directive_line);
            }

            if ln == "#![no_std]" {
                generate_needs_std_stub = false;
            }
        }

        if generate_needs_std_stub && matches!(suite, crate::common::TestSuite::CodegenLlvm) {
            lines.push(crate::directives::line::generate_needs_std(path))
        }

        Self { path, lines }
    }
}
