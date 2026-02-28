use crate::formatting::FormattingError;
use crate::{ErrorKind, FormatReport};
use annotate_snippets::{Annotation, Level, Renderer, Snippet};
use std::fmt::{self, Display};

/// A builder for [`FormatReportFormatter`].
pub struct FormatReportFormatterBuilder<'a> {
    report: &'a FormatReport,
    enable_colors: bool,
}

impl<'a> FormatReportFormatterBuilder<'a> {
    /// Creates a new [`FormatReportFormatterBuilder`].
    pub fn new(report: &'a FormatReport) -> Self {
        Self {
            report,
            enable_colors: false,
        }
    }

    /// Enables colors and formatting in the output.
    #[must_use]
    pub fn enable_colors(self, enable_colors: bool) -> Self {
        Self {
            enable_colors,
            ..self
        }
    }

    /// Creates a new [`FormatReportFormatter`] from the settings in this builder.
    pub fn build(self) -> FormatReportFormatter<'a> {
        FormatReportFormatter {
            report: self.report,
            enable_colors: self.enable_colors,
        }
    }
}

/// Formats the warnings/errors in a [`FormatReport`].
///
/// Can be created using a [`FormatReportFormatterBuilder`].
pub struct FormatReportFormatter<'a> {
    report: &'a FormatReport,
    enable_colors: bool,
}

impl<'a> Display for FormatReportFormatter<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let errors_by_file = &self.report.internal.borrow().0;

        let renderer = if self.enable_colors {
            Renderer::styled()
        } else {
            Renderer::plain()
        };

        for (file, errors) in errors_by_file {
            for error in errors {
                let error_kind = error.kind.to_string();
                let mut message =
                    error_kind_to_snippet_annotation_level(&error.kind).title(&error_kind);
                if error.is_internal() {
                    message = message.id("internal");
                }

                let message_suffix = error.msg_suffix();
                if !message_suffix.is_empty() {
                    message = message.footer(Level::Note.title(&message_suffix));
                }

                let origin = format!("{}:{}", file, error.line);
                let snippet = Snippet::source(&error.line_buffer)
                    .line_start(error.line)
                    .origin(&origin)
                    .fold(false)
                    .annotations(annotation(error));
                message = message.snippet(snippet);

                writeln!(f, "{}\n", renderer.render(message))?;
            }
        }

        if !errors_by_file.is_empty() {
            let label = format!(
                "rustfmt has failed to format. See previous {} errors.",
                self.report.warning_count()
            );
            let message = Level::Warning.title(&label);
            writeln!(f, "{}", renderer.render(message))?;
        }

        Ok(())
    }
}

fn annotation(error: &FormattingError) -> Option<Annotation<'_>> {
    let (range_start, range_length) = error.format_len();
    let range_end = range_start + range_length;

    if range_length > 0 {
        Some(Level::Error.span(range_start..range_end))
    } else {
        None
    }
}

fn error_kind_to_snippet_annotation_level(error_kind: &ErrorKind) -> Level {
    match error_kind {
        ErrorKind::LineOverflow(..)
        | ErrorKind::TrailingWhitespace
        | ErrorKind::IoError(_)
        | ErrorKind::ModuleResolutionError(_)
        | ErrorKind::ParseError
        | ErrorKind::LostComment
        | ErrorKind::BadAttr
        | ErrorKind::InvalidGlobPattern(_)
        | ErrorKind::VersionMismatch => Level::Error,
        ErrorKind::DeprecatedAttr => Level::Warning,
    }
}
