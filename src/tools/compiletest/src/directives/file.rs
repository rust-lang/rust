use camino::Utf8Path;

use crate::directives::line::{DirectiveLine, line_directive};

pub(crate) struct FileDirectives<'a> {
    pub(crate) path: &'a Utf8Path,
    pub(crate) lines: Vec<DirectiveLine<'a>>,
}

impl<'a> FileDirectives<'a> {
    pub(crate) fn from_file_contents(path: &'a Utf8Path, file_contents: &'a str) -> Self {
        let mut lines = vec![];

        for (line_number, ln) in (1..).zip(file_contents.lines()) {
            let ln = ln.trim();

            if let Some(directive_line) = line_directive(path, line_number, ln) {
                lines.push(directive_line);
            }
        }

        Self { path, lines }
    }
}
