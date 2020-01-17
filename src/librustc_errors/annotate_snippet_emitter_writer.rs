//! Emit diagnostics using the `annotate-snippets` library
//!
//! This is the equivalent of `./emitter.rs` but making use of the
//! [`annotate-snippets`][annotate_snippets] library instead of building the output ourselves.
//!
//! [annotate_snippets]: https://docs.rs/crate/annotate-snippets/

use crate::emitter::FileWithAnnotatedLines;
use crate::snippet::Line;
use crate::{CodeSuggestion, Diagnostic, DiagnosticId, Emitter, Level, SubDiagnostic};
use annotate_snippets::display_list::DisplayList;
use annotate_snippets::formatter::DisplayListFormatter;
use annotate_snippets::snippet::*;
use rustc_data_structures::sync::Lrc;
use rustc_span::source_map::SourceMap;
use rustc_span::{Loc, MultiSpan, SourceFile};

/// Generates diagnostics using annotate-snippet
pub struct AnnotateSnippetEmitterWriter {
    source_map: Option<Lrc<SourceMap>>,
    /// If true, hides the longer explanation text
    short_message: bool,
    /// If true, will normalize line numbers with `LL` to prevent noise in UI test diffs.
    ui_testing: bool,

    external_macro_backtrace: bool,
}

impl Emitter for AnnotateSnippetEmitterWriter {
    /// The entry point for the diagnostics generation
    fn emit_diagnostic(&mut self, diag: &Diagnostic) {
        let mut children = diag.children.clone();
        let (mut primary_span, suggestions) = self.primary_span_formatted(&diag);

        self.fix_multispans_in_std_macros(
            &self.source_map,
            &mut primary_span,
            &mut children,
            &diag.level,
            self.external_macro_backtrace,
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

/// Collects all the data needed to generate the data structures needed for the
/// `annotate-snippets` library.
struct DiagnosticConverter<'a> {
    source_map: Option<Lrc<SourceMap>>,
    level: Level,
    message: String,
    code: Option<DiagnosticId>,
    msp: MultiSpan,
    #[allow(dead_code)]
    children: &'a [SubDiagnostic],
    #[allow(dead_code)]
    suggestions: &'a [CodeSuggestion],
}

impl<'a> DiagnosticConverter<'a> {
    /// Turns rustc Diagnostic information into a `annotate_snippets::snippet::Snippet`.
    fn to_annotation_snippet(&self) -> Option<Snippet> {
        if let Some(source_map) = &self.source_map {
            // Make sure our primary file comes first
            let primary_lo = if let Some(ref primary_span) = self.msp.primary_span().as_ref() {
                source_map.lookup_char_pos(primary_span.lo())
            } else {
                // FIXME(#59346): Not sure when this is the case and what
                // should be done if it happens
                return None;
            };
            let annotated_files =
                FileWithAnnotatedLines::collect_annotations(&self.msp, &self.source_map);
            let slices = self.slices_for_files(annotated_files, primary_lo);

            Some(Snippet {
                title: Some(Annotation {
                    label: Some(self.message.to_string()),
                    id: self.code.clone().map(|c| match c {
                        DiagnosticId::Error(val) | DiagnosticId::Lint(val) => val,
                    }),
                    annotation_type: Self::annotation_type_for_level(self.level),
                }),
                footer: vec![],
                slices,
            })
        } else {
            // FIXME(#59346): Is it ok to return None if there's no source_map?
            None
        }
    }

    fn slices_for_files(
        &self,
        annotated_files: Vec<FileWithAnnotatedLines>,
        primary_lo: Loc,
    ) -> Vec<Slice> {
        // FIXME(#64205): Provide a test case where `annotated_files` is > 1
        annotated_files
            .iter()
            .flat_map(|annotated_file| {
                annotated_file
                    .lines
                    .iter()
                    .map(|line| {
                        let line_source = Self::source_string(annotated_file.file.clone(), &line);
                        Slice {
                            source: line_source,
                            line_start: line.line_index,
                            origin: Some(primary_lo.file.name.to_string()),
                            // FIXME(#59346): Not really sure when `fold` should be true or false
                            fold: false,
                            annotations: line
                                .annotations
                                .iter()
                                .map(|a| self.annotation_to_source_annotation(a.clone()))
                                .collect(),
                        }
                    })
                    .collect::<Vec<Slice>>()
            })
            .collect::<Vec<Slice>>()
    }

    /// Turns a `crate::snippet::Annotation` into a `SourceAnnotation`
    fn annotation_to_source_annotation(
        &self,
        annotation: crate::snippet::Annotation,
    ) -> SourceAnnotation {
        SourceAnnotation {
            range: (annotation.start_col, annotation.end_col),
            label: annotation.label.unwrap_or("".to_string()),
            annotation_type: Self::annotation_type_for_level(self.level),
        }
    }

    /// Provides the source string for the given `line` of `file`
    fn source_string(file: Lrc<SourceFile>, line: &Line) -> String {
        file.get_line(line.line_index - 1).map(|a| a.to_string()).unwrap_or(String::new())
    }

    /// Maps `Diagnostic::Level` to `snippet::AnnotationType`
    fn annotation_type_for_level(level: Level) -> AnnotationType {
        match level {
            Level::Bug | Level::Fatal | Level::Error => AnnotationType::Error,
            Level::Warning => AnnotationType::Warning,
            Level::Note => AnnotationType::Note,
            Level::Help => AnnotationType::Help,
            // FIXME(#59346): Not sure how to map these two levels
            Level::Cancelled | Level::FailureNote => AnnotationType::Error,
        }
    }
}

impl AnnotateSnippetEmitterWriter {
    pub fn new(
        source_map: Option<Lrc<SourceMap>>,
        short_message: bool,
        external_macro_backtrace: bool,
    ) -> Self {
        Self { source_map, short_message, ui_testing: false, external_macro_backtrace }
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
        children: &[SubDiagnostic],
        suggestions: &[CodeSuggestion],
    ) {
        let converter = DiagnosticConverter {
            source_map: self.source_map.clone(),
            level: *level,
            message,
            code: code.clone(),
            msp: msp.clone(),
            children,
            suggestions,
        };
        if let Some(snippet) = converter.to_annotation_snippet() {
            let dl = DisplayList::from(snippet);
            let dlf = DisplayListFormatter::new(true, self.ui_testing);
            // FIXME(#59346): Figure out if we can _always_ print to stderr or not.
            // `emitter.rs` has the `Destination` enum that lists various possible output
            // destinations.
            eprintln!("{}", dlf.format(&dl));
        };
    }
}
