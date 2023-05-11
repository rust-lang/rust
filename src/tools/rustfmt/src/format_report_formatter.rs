use crate::formatting::FormattingError;
use crate::{ErrorKind, FormatReport};
use annotate_snippets::display_list::{DisplayList, FormatOptions};
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

        let opt = FormatOptions {
            color: self.enable_colors,
            ..Default::default()
        };

        for (file, errors) in errors_by_file {
            for error in errors {
                let error_kind = error.kind.to_string();
                let title = Some(Annotation {
                    id: if error.is_internal() {
                        Some("internal")
                    } else {
                        None
                    },
                    label: Some(&error_kind),
                    annotation_type: error_kind_to_snippet_annotation_type(&error.kind),
                });

                let message_suffix = error.msg_suffix();
                let footer = if !message_suffix.is_empty() {
                    Some(Annotation {
                        id: None,
                        label: Some(message_suffix),
                        annotation_type: AnnotationType::Note,
                    })
                } else {
                    None
                };

                let origin = format!("{}:{}", file, error.line);
                let slice = Slice {
                    source: &error.line_buffer.clone(),
                    line_start: error.line,
                    origin: Some(origin.as_str()),
                    fold: false,
                    annotations: slice_annotation(error).into_iter().collect(),
                };

                let snippet = Snippet {
                    title,
                    footer: footer.into_iter().collect(),
                    slices: vec![slice],
                    opt,
                };
                writeln!(f, "{}\n", DisplayList::from(snippet))?;
            }
        }

        if !errors_by_file.is_empty() {
            let label = format!(
                "rustfmt has failed to format. See previous {} errors.",
                self.report.warning_count()
            );
            let snippet = Snippet {
                title: Some(Annotation {
                    id: None,
                    label: Some(&label),
                    annotation_type: AnnotationType::Warning,
                }),
                footer: Vec::new(),
                slices: Vec::new(),
                opt,
            };
            writeln!(f, "{}", DisplayList::from(snippet))?;
        }

        Ok(())
    }
}

fn slice_annotation(error: &FormattingError) -> Option<SourceAnnotation<'_>> {
    let (range_start, range_length) = error.format_len();
    let range_end = range_start + range_length;

    if range_length > 0 {
        Some(SourceAnnotation {
            annotation_type: AnnotationType::Error,
            range: (range_start, range_end),
            label: "",
        })
    } else {
        None
    }
}

fn error_kind_to_snippet_annotation_type(error_kind: &ErrorKind) -> AnnotationType {
    match error_kind {
        ErrorKind::LineOverflow(..)
        | ErrorKind::TrailingWhitespace
        | ErrorKind::IoError(_)
        | ErrorKind::ModuleResolutionError(_)
        | ErrorKind::ParseError
        | ErrorKind::LostComment
        | ErrorKind::BadAttr
        | ErrorKind::InvalidGlobPattern(_)
        | ErrorKind::VersionMismatch => AnnotationType::Error,
        ErrorKind::DeprecatedAttr => AnnotationType::Warning,
    }
}
