//! Emit diagnostics using the `annotate-snippets` library
//!
//! This is the equivalent of `./emitter.rs` but making use of the
//! [`annotate-snippets`][annotate_snippets] library instead of building the output ourselves.
//!
//! [annotate_snippets]: https://docs.rs/crate/annotate-snippets/

use std::borrow::Cow;
use std::error::Report;
use std::fmt::Debug;
use std::io;
use std::io::Write;
use std::sync::Arc;

use annotate_snippets::renderer::DEFAULT_TERM_WIDTH;
use annotate_snippets::{AnnotationKind, Group, Origin, Padding, Patch, Renderer, Snippet};
use anstream::ColorChoice;
use derive_setters::Setters;
use rustc_data_structures::sync::IntoDynSyncSend;
use rustc_error_messages::{FluentArgs, SpanLabel};
use rustc_lint_defs::pluralize;
use rustc_span::source_map::SourceMap;
use rustc_span::{BytePos, FileName, Pos, SourceFile, Span};
use tracing::debug;

use crate::emitter::{
    ConfusionType, Destination, MAX_SUGGESTIONS, OutputTheme, detect_confusion_type, is_different,
    normalize_whitespace, should_show_source_code,
};
use crate::registry::Registry;
use crate::translation::{Translator, to_fluent_args};
use crate::{
    CodeSuggestion, DiagInner, DiagMessage, Emitter, ErrCode, Level, MultiSpan, Style, Subdiag,
    SuggestionStyle, TerminalUrl,
};

/// Generates diagnostics using annotate-snippet
#[derive(Setters)]
pub struct AnnotateSnippetEmitter {
    #[setters(skip)]
    dst: IntoDynSyncSend<Destination>,
    sm: Option<Arc<SourceMap>>,
    #[setters(skip)]
    translator: Translator,
    short_message: bool,
    ui_testing: bool,
    ignored_directories_in_source_blocks: Vec<String>,
    diagnostic_width: Option<usize>,

    macro_backtrace: bool,
    track_diagnostics: bool,
    terminal_url: TerminalUrl,
    theme: OutputTheme,
}

impl Debug for AnnotateSnippetEmitter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnnotateSnippetEmitter")
            .field("short_message", &self.short_message)
            .field("ui_testing", &self.ui_testing)
            .field(
                "ignored_directories_in_source_blocks",
                &self.ignored_directories_in_source_blocks,
            )
            .field("diagnostic_width", &self.diagnostic_width)
            .field("macro_backtrace", &self.macro_backtrace)
            .field("track_diagnostics", &self.track_diagnostics)
            .field("terminal_url", &self.terminal_url)
            .field("theme", &self.theme)
            .finish()
    }
}

impl Emitter for AnnotateSnippetEmitter {
    /// The entry point for the diagnostics generation
    fn emit_diagnostic(&mut self, mut diag: DiagInner, _registry: &Registry) {
        let fluent_args = to_fluent_args(diag.args.iter());

        if self.track_diagnostics && diag.span.has_primary_spans() && !diag.span.is_dummy() {
            diag.children.insert(0, diag.emitted_at_sub_diag());
        }

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
            suggestions,
        );
    }

    fn source_map(&self) -> Option<&SourceMap> {
        self.sm.as_deref()
    }

    fn should_show_explain(&self) -> bool {
        !self.short_message
    }

    fn translator(&self) -> &Translator {
        &self.translator
    }

    fn supports_color(&self) -> bool {
        false
    }
}

fn annotation_level_for_level(level: Level) -> annotate_snippets::level::Level<'static> {
    match level {
        Level::Bug | Level::DelayedBug => {
            annotate_snippets::Level::ERROR.with_name("error: internal compiler error")
        }
        Level::Fatal | Level::Error => annotate_snippets::level::ERROR,
        Level::ForceWarning | Level::Warning => annotate_snippets::Level::WARNING,
        Level::Note | Level::OnceNote => annotate_snippets::Level::NOTE,
        Level::Help | Level::OnceHelp => annotate_snippets::Level::HELP,
        Level::FailureNote => annotate_snippets::Level::NOTE.no_name(),
        Level::Allow => panic!("Should not call with Allow"),
        Level::Expect => panic!("Should not call with Expect"),
    }
}

impl AnnotateSnippetEmitter {
    pub fn new(dst: Destination, translator: Translator) -> Self {
        Self {
            dst: IntoDynSyncSend(dst),
            sm: None,
            translator,
            short_message: false,
            ui_testing: false,
            ignored_directories_in_source_blocks: Vec::new(),
            diagnostic_width: None,
            macro_backtrace: false,
            track_diagnostics: false,
            terminal_url: TerminalUrl::No,
            theme: OutputTheme::Ascii,
        }
    }

    fn emit_messages_default(
        &mut self,
        level: &Level,
        msgs: &[(DiagMessage, Style)],
        args: &FluentArgs<'_>,
        code: &Option<ErrCode>,
        msp: &MultiSpan,
        children: &[Subdiag],
        suggestions: Vec<CodeSuggestion>,
    ) {
        let renderer = self.renderer();
        let annotation_level = annotation_level_for_level(*level);

        // If at least one portion of the message is styled, we need to
        // "pre-style" the message
        let mut title = if msgs.iter().any(|(_, style)| style != &crate::Style::NoStyle) {
            annotation_level
                .clone()
                .secondary_title(Cow::Owned(self.pre_style_msgs(msgs, *level, args)))
        } else {
            annotation_level.clone().primary_title(self.translator.translate_messages(msgs, args))
        };

        if let Some(c) = code {
            title = title.id(c.to_string());
            if let TerminalUrl::Yes = self.terminal_url {
                title = title.id_url(format!("https://doc.rust-lang.org/error_codes/{c}.html"));
            }
        }

        let mut report = vec![];
        let mut group = Group::with_title(title);

        // If we don't have span information, emit and exit
        let Some(sm) = self.sm.as_ref() else {
            group = group.elements(children.iter().map(|c| {
                let msg = self.translator.translate_messages(&c.messages, args).to_string();
                let level = annotation_level_for_level(c.level);
                level.message(msg)
            }));

            report.push(group);
            if let Err(e) = emit_to_destination(
                renderer.render(&report),
                level,
                &mut self.dst,
                self.short_message,
            ) {
                panic!("failed to emit error: {e}");
            }
            return;
        };

        let mut file_ann = collect_annotations(args, msp, sm, &self.translator);

        // Make sure our primary file comes first
        let primary_span = msp.primary_span().unwrap_or_default();
        if !primary_span.is_dummy() {
            let primary_lo = sm.lookup_char_pos(primary_span.lo());
            if let Ok(pos) = file_ann.binary_search_by(|(f, _)| f.name.cmp(&primary_lo.file.name)) {
                file_ann.swap(0, pos);
            }

            for (file_idx, (file, annotations)) in file_ann.into_iter().enumerate() {
                if should_show_source_code(&self.ignored_directories_in_source_blocks, sm, &file) {
                    if let Some(snippet) = self.annotated_snippet(annotations, &file.name, sm) {
                        group = group.element(snippet);
                    }
                // we can't annotate anything if the source is unavailable.
                } else if !self.short_message {
                    // We'll just print unannotated messages
                    group = self.unannotated_messages(
                        annotations,
                        &file.name,
                        sm,
                        file_idx,
                        &mut report,
                        group,
                        &annotation_level,
                    );
                    // If this is the last annotation for a file, and
                    // this is the last file, and the first child is a
                    // "secondary" message, we need to add padding
                    // ╭▸ /rustc/FAKE_PREFIX/library/core/src/clone.rs:236:13
                    // │
                    // ├ note: the late bound lifetime parameter
                    // │ (<- It adds *this*)
                    // ╰ warning: this was previously accepted
                    if let Some(c) = children.first()
                        && (!c.span.has_primary_spans() && !c.span.has_span_labels())
                    {
                        group = group.element(Padding);
                    }
                }
            }
        }

        for c in children {
            let level = annotation_level_for_level(c.level);

            // If at least one portion of the message is styled, we need to
            // "pre-style" the message
            let msg = if c.messages.iter().any(|(_, style)| style != &crate::Style::NoStyle) {
                Cow::Owned(self.pre_style_msgs(&c.messages, c.level, args))
            } else {
                self.translator.translate_messages(&c.messages, args)
            };

            // This is a secondary message with no span info
            if !c.span.has_primary_spans() && !c.span.has_span_labels() {
                group = group.element(level.clone().message(msg));
                continue;
            }

            report.push(std::mem::replace(
                &mut group,
                Group::with_title(level.clone().secondary_title(msg)),
            ));

            let mut file_ann = collect_annotations(args, &c.span, sm, &self.translator);
            let primary_span = c.span.primary_span().unwrap_or_default();
            if !primary_span.is_dummy() {
                let primary_lo = sm.lookup_char_pos(primary_span.lo());
                if let Ok(pos) =
                    file_ann.binary_search_by(|(f, _)| f.name.cmp(&primary_lo.file.name))
                {
                    file_ann.swap(0, pos);
                }
            }

            for (file_idx, (file, annotations)) in file_ann.into_iter().enumerate() {
                if should_show_source_code(&self.ignored_directories_in_source_blocks, sm, &file) {
                    if let Some(snippet) = self.annotated_snippet(annotations, &file.name, sm) {
                        group = group.element(snippet);
                    }
                // we can't annotate anything if the source is unavailable.
                } else if !self.short_message {
                    // We'll just print unannotated messages
                    group = self.unannotated_messages(
                        annotations,
                        &file.name,
                        sm,
                        file_idx,
                        &mut report,
                        group,
                        &level,
                    );
                }
            }
        }

        let suggestions_expected = suggestions
            .iter()
            .filter(|s| {
                matches!(
                    s.style,
                    SuggestionStyle::HideCodeInline
                        | SuggestionStyle::ShowCode
                        | SuggestionStyle::ShowAlways
                )
            })
            .count();
        for suggestion in suggestions {
            match suggestion.style {
                SuggestionStyle::CompletelyHidden => {
                    // do not display this suggestion, it is meant only for tools
                }
                SuggestionStyle::HideCodeAlways => {
                    let msg = self
                        .translator
                        .translate_messages(&[(suggestion.msg.to_owned(), Style::HeaderMsg)], args);
                    group = group.element(annotate_snippets::Level::HELP.message(msg));
                }
                SuggestionStyle::HideCodeInline
                | SuggestionStyle::ShowCode
                | SuggestionStyle::ShowAlways => {
                    let substitutions = suggestion
                        .substitutions
                        .into_iter()
                        .filter(|subst| {
                            // Suggestions coming from macros can have malformed spans. This is a heavy
                            // handed approach to avoid ICEs by ignoring the suggestion outright.
                            let invalid =
                                subst.parts.iter().any(|item| sm.is_valid_span(item.span).is_err());
                            if invalid {
                                debug!("suggestion contains an invalid span: {:?}", subst);
                            }
                            !invalid
                        })
                        .filter_map(|mut subst| {
                            // Assumption: all spans are in the same file, and all spans
                            // are disjoint. Sort in ascending order.
                            subst.parts.sort_by_key(|part| part.span.lo());
                            // Verify the assumption that all spans are disjoint
                            debug_assert_eq!(
                                subst.parts.array_windows().find(|[a, b]| a.span.overlaps(b.span)),
                                None,
                                "all spans must be disjoint",
                            );

                            // Account for cases where we are suggesting the same code that's already
                            // there. This shouldn't happen often, but in some cases for multipart
                            // suggestions it's much easier to handle it here than in the origin.
                            subst.parts.retain(|p| is_different(sm, &p.snippet, p.span));

                            let item_span = subst.parts.first()?;
                            let file = sm.lookup_source_file(item_span.span.lo());
                            if should_show_source_code(
                                &self.ignored_directories_in_source_blocks,
                                sm,
                                &file,
                            ) {
                                Some(subst)
                            } else {
                                None
                            }
                        })
                        .collect::<Vec<_>>();

                    if substitutions.is_empty() {
                        continue;
                    }
                    let mut msg = self
                        .translator
                        .translate_message(&suggestion.msg, args)
                        .map_err(Report::new)
                        .unwrap()
                        .to_string();

                    let lo = substitutions
                        .iter()
                        .find_map(|sub| sub.parts.first().map(|p| p.span.lo()))
                        .unwrap();
                    let file = sm.lookup_source_file(lo);

                    let filename =
                        sm.filename_for_diagnostics(&file.name).to_string_lossy().to_string();

                    let other_suggestions = substitutions.len().saturating_sub(MAX_SUGGESTIONS);

                    let subs = substitutions
                        .into_iter()
                        .take(MAX_SUGGESTIONS)
                        .filter_map(|sub| {
                            let mut confusion_type = ConfusionType::None;
                            for part in &sub.parts {
                                let part_confusion =
                                    detect_confusion_type(sm, &part.snippet, part.span);
                                confusion_type = confusion_type.combine(part_confusion);
                            }

                            if !matches!(confusion_type, ConfusionType::None) {
                                msg.push_str(confusion_type.label_text());
                            }

                            let mut parts = sub
                                .parts
                                .into_iter()
                                .filter_map(|p| {
                                    if is_different(sm, &p.snippet, p.span) {
                                        Some((p.span, p.snippet))
                                    } else {
                                        None
                                    }
                                })
                                .collect::<Vec<_>>();

                            if parts.is_empty() {
                                None
                            } else {
                                let spans = parts.iter().map(|(span, _)| *span).collect::<Vec<_>>();
                                // The suggestion adds an entire line of code, ending on a newline, so we'll also
                                // print the *following* line, to provide context of what we're advising people to
                                // do. Otherwise you would only see contextless code that can be confused for
                                // already existing code, despite the colors and UI elements.
                                // We special case `#[derive(_)]\n` and other attribute suggestions, because those
                                // are the ones where context is most useful.
                                let fold = if let [(p, snippet)] = &mut parts[..]
                                    && snippet.trim().starts_with("#[")
                                    // This allows for spaces to come between the attribute and the newline
                                    && snippet.trim().ends_with("]")
                                    && snippet.ends_with('\n')
                                    && p.hi() == p.lo()
                                    && let Ok(b) = sm.span_to_prev_source(*p)
                                    && let b = b.rsplit_once('\n').unwrap_or_else(|| ("", &b)).1
                                    && b.trim().is_empty()
                                {
                                    // FIXME: This is a hack:
                                    // The span for attribute suggestions often times points to the
                                    // beginning of an item, disregarding leading whitespace. This
                                    // causes the attribute to be properly indented, but leaves original
                                    // item without indentation when rendered.
                                    // This fixes that problem by adjusting the span to point to the start
                                    // of the whitespace, and adds the whitespace to the replacement.
                                    //
                                    // Source: "    extern "custom" fn negate(a: i64) -> i64 {\n"
                                    // Span: 4..4
                                    // Replacement: "#[unsafe(naked)]\n"
                                    //
                                    // Before:
                                    // help: convert this to an `#[unsafe(naked)]` function
                                    //    |
                                    // LL +     #[unsafe(naked)]
                                    // LL | extern "custom" fn negate(a: i64) -> i64 {
                                    //    |
                                    //
                                    // After
                                    // help: convert this to an `#[unsafe(naked)]` function
                                    //    |
                                    // LL +     #[unsafe(naked)]
                                    // LL |     extern "custom" fn negate(a: i64) -> i64 {
                                    //    |
                                    if !b.is_empty() && !snippet.ends_with(b) {
                                        snippet.insert_str(0, b);
                                        let offset = BytePos(b.len() as u32);
                                        *p = p.with_lo(p.lo() - offset).shrink_to_lo();
                                    }
                                    false
                                } else {
                                    true
                                };

                                if let Some((bounding_span, source, line_offset)) =
                                    shrink_file(spans.as_slice(), &file.name, sm)
                                {
                                    let adj_lo = bounding_span.lo().to_usize();
                                    Some(
                                        Snippet::source(source)
                                            .line_start(line_offset)
                                            .path(filename.clone())
                                            .fold(fold)
                                            .patches(parts.into_iter().map(
                                                |(span, replacement)| {
                                                    let lo =
                                                        span.lo().to_usize().saturating_sub(adj_lo);
                                                    let hi =
                                                        span.hi().to_usize().saturating_sub(adj_lo);

                                                    Patch::new(lo..hi, replacement)
                                                },
                                            )),
                                    )
                                } else {
                                    None
                                }
                            }
                        })
                        .collect::<Vec<_>>();
                    if !subs.is_empty() {
                        report.push(std::mem::replace(
                            &mut group,
                            Group::with_title(annotate_snippets::Level::HELP.secondary_title(msg)),
                        ));

                        group = group.elements(subs);
                        if other_suggestions > 0 {
                            group = group.element(
                                annotate_snippets::Level::NOTE.no_name().message(format!(
                                    "and {} other candidate{}",
                                    other_suggestions,
                                    pluralize!(other_suggestions)
                                )),
                            );
                        }
                    }
                }
            }
        }

        // FIXME: This hack should be removed once annotate_snippets is the
        // default emitter.
        if suggestions_expected > 0 && report.is_empty() {
            group = group.element(Padding);
        }

        if !group.is_empty() {
            report.push(group);
        }
        if let Err(e) =
            emit_to_destination(renderer.render(&report), level, &mut self.dst, self.short_message)
        {
            panic!("failed to emit error: {e}");
        }
    }

    fn renderer(&self) -> Renderer {
        let width = if let Some(width) = self.diagnostic_width {
            width
        } else if self.ui_testing || cfg!(miri) {
            DEFAULT_TERM_WIDTH
        } else {
            termize::dimensions().map(|(w, _)| w).unwrap_or(DEFAULT_TERM_WIDTH)
        };
        let decor_style = match self.theme {
            OutputTheme::Ascii => annotate_snippets::renderer::DecorStyle::Ascii,
            OutputTheme::Unicode => annotate_snippets::renderer::DecorStyle::Unicode,
        };

        match self.dst.current_choice() {
            ColorChoice::AlwaysAnsi | ColorChoice::Always | ColorChoice::Auto => Renderer::styled(),
            ColorChoice::Never => Renderer::plain(),
        }
        .term_width(width)
        .anonymized_line_numbers(self.ui_testing)
        .decor_style(decor_style)
        .short_message(self.short_message)
    }

    fn pre_style_msgs(
        &self,
        msgs: &[(DiagMessage, Style)],
        level: Level,
        args: &FluentArgs<'_>,
    ) -> String {
        msgs.iter()
            .filter_map(|(m, style)| {
                let text = self.translator.translate_message(m, args).map_err(Report::new).unwrap();
                let style = style.anstyle(level);
                if text.is_empty() { None } else { Some(format!("{style}{text}{style:#}")) }
            })
            .collect()
    }

    fn annotated_snippet<'a>(
        &self,
        annotations: Vec<Annotation>,
        file_name: &FileName,
        sm: &Arc<SourceMap>,
    ) -> Option<Snippet<'a, annotate_snippets::Annotation<'a>>> {
        let spans = annotations.iter().map(|a| a.span).collect::<Vec<_>>();
        if let Some((bounding_span, source, offset_line)) = shrink_file(&spans, file_name, sm) {
            let adj_lo = bounding_span.lo().to_usize();
            let filename = sm.filename_for_diagnostics(file_name).to_string_lossy().to_string();
            Some(Snippet::source(source).line_start(offset_line).path(filename).annotations(
                annotations.into_iter().map(move |a| {
                    let lo = a.span.lo().to_usize().saturating_sub(adj_lo);
                    let hi = a.span.hi().to_usize().saturating_sub(adj_lo);
                    let ann = a.kind.span(lo..hi);
                    if let Some(label) = a.label { ann.label(label) } else { ann }
                }),
            ))
        } else {
            None
        }
    }

    fn unannotated_messages<'a>(
        &self,
        annotations: Vec<Annotation>,
        file_name: &FileName,
        sm: &Arc<SourceMap>,
        file_idx: usize,
        report: &mut Vec<Group<'a>>,
        mut group: Group<'a>,
        level: &annotate_snippets::level::Level<'static>,
    ) -> Group<'a> {
        let filename = sm.filename_for_diagnostics(file_name).to_string_lossy().to_string();
        let mut line_tracker = vec![];
        for (i, a) in annotations.into_iter().enumerate() {
            let lo = sm.lookup_char_pos(a.span.lo());
            let hi = sm.lookup_char_pos(a.span.hi());
            if i == 0 || (a.label.is_some()) {
                // Render each new file after the first in its own Group
                //    ╭▸ $DIR/deriving-meta-unknown-trait.rs:1:10
                //    │
                // LL │ #[derive(Eqr)]
                //    │          ━━━
                //    ╰╴ (<- It makes it so *this* will get printed)
                //    ╭▸ $SRC_DIR/core/src/option.rs:594:0
                //    ⸬  $SRC_DIR/core/src/option.rs:602:4
                //    │
                //    ╰ note: not covered
                if i == 0 && file_idx != 0 {
                    report.push(std::mem::replace(&mut group, Group::with_level(level.clone())));
                }

                if !line_tracker.contains(&lo.line) {
                    line_tracker.push(lo.line);
                    // ╭▸ $SRC_DIR/core/src/option.rs:594:0 (<- It adds *this*)
                    // ⸬  $SRC_DIR/core/src/option.rs:602:4
                    // │
                    // ╰ note: not covered
                    group = group.element(
                        Origin::path(filename.clone())
                            .line(sm.doctest_offset_line(file_name, lo.line))
                            .char_column(lo.col_display),
                    );
                }

                if hi.line > lo.line
                    && a.label.as_ref().is_some_and(|l| !l.is_empty())
                    && !line_tracker.contains(&hi.line)
                {
                    line_tracker.push(hi.line);
                    // ╭▸ $SRC_DIR/core/src/option.rs:594:0
                    // ⸬  $SRC_DIR/core/src/option.rs:602:4 (<- It adds *this*)
                    // │
                    // ╰ note: not covered
                    group = group.element(
                        Origin::path(filename.clone())
                            .line(sm.doctest_offset_line(file_name, hi.line))
                            .char_column(hi.col_display),
                    );
                }

                if let Some(label) = a.label
                    && !label.is_empty()
                {
                    // ╭▸ $SRC_DIR/core/src/option.rs:594:0
                    // ⸬  $SRC_DIR/core/src/option.rs:602:4
                    // │ (<- It adds *this*)
                    // ╰ note: not covered (<- and *this*)
                    group = group
                        .element(Padding)
                        .element(annotate_snippets::Level::NOTE.message(label));
                }
            }
        }
        group
    }
}

fn emit_to_destination(
    rendered: String,
    lvl: &Level,
    dst: &mut Destination,
    short_message: bool,
) -> io::Result<()> {
    use crate::lock;
    let _buffer_lock = lock::acquire_global_lock("rustc_errors");
    writeln!(dst, "{rendered}")?;
    if !short_message && !lvl.is_failure_note() {
        writeln!(dst)?;
    }
    dst.flush()?;
    Ok(())
}

#[derive(Debug)]
struct Annotation {
    kind: AnnotationKind,
    span: Span,
    label: Option<String>,
}

fn collect_annotations(
    args: &FluentArgs<'_>,
    msp: &MultiSpan,
    sm: &Arc<SourceMap>,
    translator: &Translator,
) -> Vec<(Arc<SourceFile>, Vec<Annotation>)> {
    let mut output: Vec<(Arc<SourceFile>, Vec<Annotation>)> = vec![];

    for SpanLabel { span, is_primary, label } in msp.span_labels() {
        // If we don't have a useful span, pick the primary span if that exists.
        // Worst case we'll just print an error at the top of the main file.
        let span = match (span.is_dummy(), msp.primary_span()) {
            (_, None) | (false, _) => span,
            (true, Some(span)) => span,
        };
        let file = sm.lookup_source_file(span.lo());

        let kind = if is_primary { AnnotationKind::Primary } else { AnnotationKind::Context };

        let label = label.as_ref().map(|m| {
            normalize_whitespace(
                &translator.translate_message(m, args).map_err(Report::new).unwrap(),
            )
        });

        let ann = Annotation { kind, span, label };
        if sm.is_valid_span(ann.span).is_ok() {
            // Look through each of our files for the one we're adding to. We
            // use each files `stable_id` to avoid issues with file name
            // collisions when multiple versions of the same crate are present
            // in the dependency graph
            if let Some((_, annotations)) =
                output.iter_mut().find(|(f, _)| f.stable_id == file.stable_id)
            {
                annotations.push(ann);
            } else {
                output.push((file, vec![ann]));
            }
        }
    }
    output
}

fn shrink_file(
    spans: &[Span],
    file_name: &FileName,
    sm: &Arc<SourceMap>,
) -> Option<(Span, String, usize)> {
    let lo_byte = spans.iter().map(|s| s.lo()).min()?;
    let lo_loc = sm.lookup_char_pos(lo_byte);
    let lo = lo_loc.file.line_bounds(lo_loc.line.saturating_sub(1)).start;

    let hi_byte = spans.iter().map(|s| s.hi()).max()?;
    let hi_loc = sm.lookup_char_pos(hi_byte);
    let hi = lo_loc.file.line_bounds(hi_loc.line.saturating_sub(1)).end;

    let bounding_span = Span::with_root_ctxt(lo, hi);
    let source = sm.span_to_snippet(bounding_span).unwrap_or_default();
    let offset_line = sm.doctest_offset_line(file_name, lo_loc.line);

    Some((bounding_span, source, offset_line))
}
