//! Emit diagnostics using the `annotate-snippets` library
//!
//! This is the equivalent of `./emitter.rs` but making use of the
//! [`annotate-snippets`][annotate_snippets] library instead of building the output ourselves.
//!
//! [annotate_snippets]: https://docs.rs/crate/annotate-snippets/

use crate::emitter::FileWithAnnotatedLines;
use crate::snippet::Line;
use crate::{CodeSuggestion, Diagnostic, DiagnosticId, Emitter, Level, SubDiagnostic};
use annotate_snippets::display_list::{DisplayList, FormatOptions};
use annotate_snippets::snippet::*;
use rustc_data_structures::sync::Lrc;
use rustc_span::source_map::SourceMap;
use rustc_span::{MultiSpan, SourceFile};

/// Generates diagnostics using annotate-snippet
pub struct AnnotateSnippetEmitterWriter {
    source_map: Option<Lrc<SourceMap>>,
    /// If true, hides the longer explanation text
    short_message: bool,
    /// If true, will normalize line numbers with `LL` to prevent noise in UI test diffs.
    ui_testing: bool,

    macro_backtrace: bool,
}

impl Emitter for AnnotateSnippetEmitterWriter {
    /// The entry point for the diagnostics generation
    fn emit_diagnostic(&mut self, diag: &Diagnostic) {
        let mut children = diag.children.clone();
        let (mut primary_span, suggestions) = self.primary_span_formatted(&diag);

        self.fix_multispans_in_extern_macros_and_render_macro_backtrace(
            &self.source_map,
            &mut primary_span,
            &mut children,
            &diag.level,
            self.macro_backtrace,
        );

        self.emit_messages_default(
            &diag.level,
            diag.message(),
            &diag.code,
            &primary_span,
            &children,
            &suggestions,
        );
    }

    fn source_map(&self) -> Option<&Lrc<SourceMap>> {
        self.source_map.as_ref()
    }

    fn should_show_explain(&self) -> bool {
        !self.short_message
    }
}

/// Provides the source string for the given `line` of `file`
fn source_string(file: Lrc<SourceFile>, line: &Line) -> String {
    file.get_line(line.line_index - 1).map(|a| a.to_string()).unwrap_or_default()
}

/// Maps `Diagnostic::Level` to `snippet::AnnotationType`
fn annotation_type_for_level(level: Level) -> AnnotationType {
    match level {
        Level::Bug | Level::Fatal | Level::Error { .. } => AnnotationType::Error,
        Level::Warning => AnnotationType::Warning,
        Level::Note => AnnotationType::Note,
        Level::Help => AnnotationType::Help,
        // FIXME(#59346): Not sure how to map these two levels
        Level::Cancelled | Level::FailureNote => AnnotationType::Error,
        Level::Allow => panic!("Should not call with Allow"),
    }
}

impl AnnotateSnippetEmitterWriter {
    pub fn new(
        source_map: Option<Lrc<SourceMap>>,
        short_message: bool,
        macro_backtrace: bool,
    ) -> Self {
        Self { source_map, short_message, ui_testing: false, macro_backtrace }
    }

    /// Allows to modify `Self` to enable or disable the `ui_testing` flag.
    ///
    /// If this is set to true, line numbers will be normalized as `LL` in the output.
    pub fn ui_testing(mut self, ui_testing: bool) -> Self {
        self.ui_testing = ui_testing;
        self
    }

    fn emit_messages_default(
        &mut self,
        level: &Level,
        message: String,
        code: &Option<DiagnosticId>,
        msp: &MultiSpan,
        _children: &[SubDiagnostic],
        _suggestions: &[CodeSuggestion],
    ) {
        if let Some(source_map) = &self.source_map {
            // Make sure our primary file comes first
            let primary_lo = if let Some(ref primary_span) = msp.primary_span().as_ref() {
                if primary_span.is_dummy() {
                    // FIXME(#59346): Not sure when this is the case and what
                    // should be done if it happens
                    return;
                } else {
                    source_map.lookup_char_pos(primary_span.lo())
                }
            } else {
                // FIXME(#59346): Not sure when this is the case and what
                // should be done if it happens
                return;
            };
            let mut annotated_files =
                FileWithAnnotatedLines::collect_annotations(msp, &self.source_map);
            if let Ok(pos) =
                annotated_files.binary_search_by(|x| x.file.name.cmp(&primary_lo.file.name))
            {
                annotated_files.swap(0, pos);
            }
            // owned: line source, line index, annotations
            type Owned = (String, usize, Vec<crate::snippet::Annotation>);
            let filename = source_map.filename_for_diagnostics(&primary_lo.file.name);
            let origin = filename.to_string_lossy();
            let annotated_files: Vec<Owned> = annotated_files
                .into_iter()
                .flat_map(|annotated_file| {
                    let file = annotated_file.file;
                    annotated_file
                        .lines
                        .into_iter()
                        .map(|line| {
                            (source_string(file.clone(), &line), line.line_index, line.annotations)
                        })
                        .collect::<Vec<Owned>>()
                })
                .collect();
            let snippet = Snippet {
                title: Some(Annotation {
                    label: Some(&message),
                    id: code.as_ref().map(|c| match c {
                        DiagnosticId::Error(val) | DiagnosticId::Lint { name: val, .. } => {
                            val.as_str()
                        }
                    }),
                    annotation_type: annotation_type_for_level(*level),
                }),
                footer: vec![],
                opt: FormatOptions { color: true, anonymized_line_numbers: self.ui_testing },
                slices: annotated_files
                    .iter()
                    .map(|(source, line_index, annotations)| {
                        Slice {
                            source,
                            line_start: *line_index,
                            origin: Some(&origin),
                            // FIXME(#59346): Not really sure when `fold` should be true or false
                            fold: false,
                            annotations: annotations
                                .iter()
                                .map(|annotation| SourceAnnotation {
                                    range: (annotation.start_col, annotation.end_col),
                                    label: annotation.label.as_deref().unwrap_or_default(),
                                    annotation_type: annotation_type_for_level(*level),
                                })
                                .collect(),
                        }
                    })
                    .collect(),
            };
            // FIXME(#59346): Figure out if we can _always_ print to stderr or not.
            // `emitter.rs` has the `Destination` enum that lists various possible output
            // destinations.
            eprintln!("{}", DisplayList::from(snippet))
        }
        // FIXME(#59346): Is it ok to return None if there's no source_map?
    }
}
