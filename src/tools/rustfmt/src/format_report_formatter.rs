use crate::formatting::FormattingError;
use crate::{ErrorKind, FormatReport};
use annotate_snippets::{Annotation, AnnotationKind, Group, Level, Padding, Renderer, Snippet};
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
                let mut title =
                    error_kind_to_snippet_annotation_level(&error.kind).primary_title(error_kind);
                if error.is_internal() {
                    title = title.id("internal");
                }
                let snippet = Snippet::source(&error.line_buffer)
                    .line_start(error.line)
                    .path(format!("{file}:{}", error.line))
                    .fold(false)
                    .annotations(annotation(error));
                let mut group = title.element(snippet);
                if let Some(message_suffix) = error.msg_suffix() {
                    group = group.element(Level::NOTE.message(message_suffix));
                } else {
                    group = group.element(Padding);
                }
                writeln!(f, "{}\n", renderer.render(&[group]))?;
            }
        }

        if !errors_by_file.is_empty() {
            let label = format!(
                "rustfmt has failed to format. See previous {} errors.",
                self.report.warning_count()
            );
            let group = Group::with_title(Level::WARNING.primary_title(label));
            writeln!(f, "{}\n", renderer.render(&[group]))?;
        }
        Ok(())
    }
}

fn annotation(error: &FormattingError) -> Option<Annotation<'_>> {
    error
        .highlight
        .clone()
        .map(|range| AnnotationKind::Primary.span(range))
}

fn error_kind_to_snippet_annotation_level(error_kind: &ErrorKind) -> Level<'_> {
    match error_kind {
        ErrorKind::LineOverflow(..)
        | ErrorKind::TrailingWhitespace
        | ErrorKind::IoError(_)
        | ErrorKind::ModuleResolutionError(_)
        | ErrorKind::ParseError
        | ErrorKind::LostComment
        | ErrorKind::BadAttr
        | ErrorKind::InvalidGlobPattern(_)
        | ErrorKind::VersionMismatch => Level::ERROR,
        ErrorKind::DeprecatedAttr => Level::WARNING,
    }
}
