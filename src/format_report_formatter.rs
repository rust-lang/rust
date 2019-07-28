use crate::config::FileName;
use crate::formatting::FormattingError;
use crate::{ErrorKind, FormatReport};
use annotate_snippets::display_list::DisplayList;
use annotate_snippets::formatter::DisplayListFormatter;
use annotate_snippets::snippet::{Annotation, AnnotationType, Slice, Snippet, SourceAnnotation};
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
        let formatter = DisplayListFormatter::new(self.enable_colors, false);
        let errors_by_file = &self.report.internal.borrow().0;

        for (file, errors) in errors_by_file {
            for error in errors {
                let snippet = formatting_error_to_snippet(file, error);
                writeln!(f, "{}\n", formatter.format(&DisplayList::from(snippet)))?;
            }
        }

        if !errors_by_file.is_empty() {
            let snippet = formatting_failure_snippet(self.report.warning_count());
            writeln!(f, "{}", formatter.format(&DisplayList::from(snippet)))?;
        }

        Ok(())
    }
}

fn formatting_failure_snippet(warning_count: usize) -> Snippet {
    Snippet {
        title: Some(Annotation {
            id: None,
            label: Some(format!(
                "rustfmt has failed to format. See previous {} errors.",
                warning_count
            )),
            annotation_type: AnnotationType::Warning,
        }),
        footer: Vec::new(),
        slices: Vec::new(),
    }
}

fn formatting_error_to_snippet(file: &FileName, error: &FormattingError) -> Snippet {
    let slices = vec![snippet_code_slice(file, error)];
    let title = Some(snippet_title(error));
    let footer = snippet_footer(error).into_iter().collect();

    Snippet {
        title,
        footer,
        slices,
    }
}

fn snippet_title(error: &FormattingError) -> Annotation {
    let annotation_type = error_kind_to_snippet_annotation_type(&error.kind);

    Annotation {
        id: title_annotation_id(error),
        label: Some(error.kind.to_string()),
        annotation_type,
    }
}

fn snippet_footer(error: &FormattingError) -> Option<Annotation> {
    let message_suffix = error.msg_suffix();

    if !message_suffix.is_empty() {
        Some(Annotation {
            id: None,
            label: Some(message_suffix.to_string()),
            annotation_type: AnnotationType::Note,
        })
    } else {
        None
    }
}

fn snippet_code_slice(file: &FileName, error: &FormattingError) -> Slice {
    let annotations = slice_annotation(error).into_iter().collect();
    let origin = Some(format!("{}:{}", file, error.line));
    let source = error.line_buffer.clone();

    Slice {
        source,
        line_start: error.line,
        origin,
        fold: false,
        annotations,
    }
}

fn slice_annotation(error: &FormattingError) -> Option<SourceAnnotation> {
    let (range_start, range_length) = error.format_len();
    let range_end = range_start + range_length;

    if range_length > 0 {
        Some(SourceAnnotation {
            annotation_type: AnnotationType::Error,
            range: (range_start, range_end),
            label: String::new(),
        })
    } else {
        None
    }
}

fn title_annotation_id(error: &FormattingError) -> Option<String> {
    const INTERNAL_ERROR_ID: &str = "internal";

    if error.is_internal() {
        Some(INTERNAL_ERROR_ID.to_string())
    } else {
        None
    }
}

fn error_kind_to_snippet_annotation_type(error_kind: &ErrorKind) -> AnnotationType {
    match error_kind {
        ErrorKind::LineOverflow(..)
        | ErrorKind::TrailingWhitespace
        | ErrorKind::IoError(_)
        | ErrorKind::ParseError
        | ErrorKind::LostComment
        | ErrorKind::LicenseCheck
        | ErrorKind::BadAttr
        | ErrorKind::InvalidGlobPattern(_)
        | ErrorKind::VersionMismatch => AnnotationType::Error,
        ErrorKind::BadIssue(_) | ErrorKind::DeprecatedAttr => AnnotationType::Warning,
    }
}
