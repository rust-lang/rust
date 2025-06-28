//! Emit diagnostics using the `annotate-snippets` library
//!
//! This is the equivalent of `./emitter.rs` but making use of the
//! [`annotate-snippets`][annotate_snippets] library instead of building the output ourselves.
//!
//! [annotate_snippets]: https://docs.rs/crate/annotate-snippets/

use std::sync::Arc;

use annotate_snippets::{Renderer, Snippet};
use rustc_error_messages::FluentArgs;
use rustc_span::SourceFile;
use rustc_span::source_map::SourceMap;

use crate::emitter::FileWithAnnotatedLines;
use crate::registry::Registry;
use crate::snippet::Line;
use crate::translation::{Translator, to_fluent_args};
use crate::{
    CodeSuggestion, DiagInner, DiagMessage, Emitter, ErrCode, Level, MultiSpan, Style, Subdiag,
};

/// Generates diagnostics using annotate-snippet
pub struct AnnotateSnippetEmitter {
    source_map: Option<Arc<SourceMap>>,
    translator: Translator,

    /// If true, hides the longer explanation text
    short_message: bool,
    /// If true, will normalize line numbers with `LL` to prevent noise in UI test diffs.
    ui_testing: bool,

    macro_backtrace: bool,
}

impl Emitter for AnnotateSnippetEmitter {
    /// The entry point for the diagnostics generation
    fn emit_diagnostic(&mut self, mut diag: DiagInner, _registry: &Registry) {
        let fluent_args = to_fluent_args(diag.args.iter());

        let mut suggestions = diag.suggestions.unwrap_tag();
        self.primary_span_formatted(&mut diag.span, &mut suggestions, &fluent_args);

        self.fix_multispans_in_extern_macros_and_render_macro_backtrace(
            &mut diag.span,
            &mut diag.children,
            &diag.level,
            self.macro_backtrace,
        );

        self.emit_messages_default(
            &diag.level,
            &diag.messages,
            &fluent_args,
            &diag.code,
            &diag.span,
            &diag.children,
            &suggestions,
        );
    }

    fn source_map(&self) -> Option<&SourceMap> {
        self.source_map.as_deref()
    }

    fn should_show_explain(&self) -> bool {
        !self.short_message
    }

    fn translator(&self) -> &Translator {
        &self.translator
    }
}

/// Provides the source string for the given `line` of `file`
fn source_string(file: Arc<SourceFile>, line: &Line) -> String {
    file.get_line(line.line_index - 1).map(|a| a.to_string()).unwrap_or_default()
}

/// Maps [`crate::Level`] to [`annotate_snippets::Level`]
fn annotation_level_for_level(level: Level) -> annotate_snippets::Level {
    match level {
        Level::Bug | Level::Fatal | Level::Error | Level::DelayedBug => {
            annotate_snippets::Level::Error
        }
        Level::ForceWarning | Level::Warning => annotate_snippets::Level::Warning,
        Level::Note | Level::OnceNote => annotate_snippets::Level::Note,
        Level::Help | Level::OnceHelp => annotate_snippets::Level::Help,
        // FIXME(#59346): Not sure how to map this level
        Level::FailureNote => annotate_snippets::Level::Error,
        Level::Allow => panic!("Should not call with Allow"),
        Level::Expect => panic!("Should not call with Expect"),
    }
}

impl AnnotateSnippetEmitter {
    pub fn new(
        source_map: Option<Arc<SourceMap>>,
        translator: Translator,
        short_message: bool,
        macro_backtrace: bool,
    ) -> Self {
        Self { source_map, translator, short_message, ui_testing: false, macro_backtrace }
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
        messages: &[(DiagMessage, Style)],
        args: &FluentArgs<'_>,
        code: &Option<ErrCode>,
        msp: &MultiSpan,
        _children: &[Subdiag],
        _suggestions: &[CodeSuggestion],
    ) {
        let message = self.translator.translate_messages(messages, args);
        if let Some(source_map) = &self.source_map {
            // Make sure our primary file comes first
            let primary_lo = if let Some(primary_span) = msp.primary_span().as_ref() {
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
            let mut annotated_files = FileWithAnnotatedLines::collect_annotations(self, args, msp);
            if let Ok(pos) =
                annotated_files.binary_search_by(|x| x.file.name.cmp(&primary_lo.file.name))
            {
                annotated_files.swap(0, pos);
            }
            // owned: file name, line source, line index, annotations
            type Owned = (String, String, usize, Vec<crate::snippet::Annotation>);
            let annotated_files: Vec<Owned> = annotated_files
                .into_iter()
                .flat_map(|annotated_file| {
                    let file = annotated_file.file;
                    annotated_file
                        .lines
                        .into_iter()
                        .map(|line| {
                            // Ensure the source file is present before we try
                            // to load a string from it.
                            // FIXME(#115869): support -Z ignore-directory-in-diagnostics-source-blocks
                            source_map.ensure_source_file_source_present(&file);
                            (
                                format!("{}", source_map.filename_for_diagnostics(&file.name)),
                                source_string(Arc::clone(&file), &line),
                                line.line_index,
                                line.annotations,
                            )
                        })
                        .collect::<Vec<Owned>>()
                })
                .collect();
            let code = code.map(|code| code.to_string());

            let snippets =
                annotated_files.iter().map(|(file_name, source, line_index, annotations)| {
                    Snippet::source(source)
                        .line_start(*line_index)
                        .origin(file_name)
                        // FIXME(#59346): Not really sure when `fold` should be true or false
                        .fold(false)
                        .annotations(annotations.iter().map(|annotation| {
                            annotation_level_for_level(*level)
                                .span(annotation.start_col.display..annotation.end_col.display)
                                .label(annotation.label.as_deref().unwrap_or_default())
                        }))
                });
            let mut message = annotation_level_for_level(*level).title(&message).snippets(snippets);
            if let Some(code) = code.as_deref() {
                message = message.id(code)
            }
            // FIXME(#59346): Figure out if we can _always_ print to stderr or not.
            // `emitter.rs` has the `Destination` enum that lists various possible output
            // destinations.
            let renderer = Renderer::plain().anonymized_line_numbers(self.ui_testing);
            eprintln!("{}", renderer.render(message))
        }
        // FIXME(#59346): Is it ok to return None if there's no source_map?
    }
}
