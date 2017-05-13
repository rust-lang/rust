// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A JSON emitter for errors.
//!
//! This works by converting errors to a simplified structural format (see the
//! structs at the start of the file) and then serialising them. These should
//! contain as much information about the error as possible.
//!
//! The format of the JSON output should be considered *unstable*. For now the
//! structs at the end of this file (Diagnostic*) specify the error format.

// FIXME spec the JSON output properly.

use codemap::CodeMap;
use syntax_pos::{self, MacroBacktrace, Span, SpanLabel, MultiSpan};
use errors::registry::Registry;
use errors::{DiagnosticBuilder, SubDiagnostic, RenderSpan, CodeSuggestion, CodeMapper};
use errors::emitter::Emitter;

use std::rc::Rc;
use std::io::{self, Write};
use std::vec;

use rustc_serialize::json::as_json;

pub struct JsonEmitter {
    dst: Box<Write + Send>,
    registry: Option<Registry>,
    cm: Rc<CodeMapper + 'static>,
}

impl JsonEmitter {
    pub fn stderr(registry: Option<Registry>,
                  code_map: Rc<CodeMap>) -> JsonEmitter {
        JsonEmitter {
            dst: Box::new(io::stderr()),
            registry: registry,
            cm: code_map,
        }
    }

    pub fn basic() -> JsonEmitter {
        JsonEmitter::stderr(None, Rc::new(CodeMap::new()))
    }

    pub fn new(dst: Box<Write + Send>,
               registry: Option<Registry>,
               code_map: Rc<CodeMap>) -> JsonEmitter {
        JsonEmitter {
            dst: dst,
            registry: registry,
            cm: code_map,
        }
    }
}

impl Emitter for JsonEmitter {
    fn emit(&mut self, db: &DiagnosticBuilder) {
        let data = Diagnostic::from_diagnostic_builder(db, self);
        if let Err(e) = writeln!(&mut self.dst, "{}", as_json(&data)) {
            panic!("failed to print diagnostics: {:?}", e);
        }
    }
}

// The following data types are provided just for serialisation.

#[derive(RustcEncodable)]
struct Diagnostic<'a> {
    /// The primary error message.
    message: &'a str,
    code: Option<DiagnosticCode>,
    /// "error: internal compiler error", "error", "warning", "note", "help".
    level: &'static str,
    spans: Vec<DiagnosticSpan>,
    /// Associated diagnostic messages.
    children: Vec<Diagnostic<'a>>,
    /// The message as rustc would render it. Currently this is only
    /// `Some` for "suggestions", but eventually it will include all
    /// snippets.
    rendered: Option<String>,
}

#[derive(RustcEncodable)]
struct DiagnosticSpan {
    file_name: String,
    byte_start: u32,
    byte_end: u32,
    /// 1-based.
    line_start: usize,
    line_end: usize,
    /// 1-based, character offset.
    column_start: usize,
    column_end: usize,
    /// Is this a "primary" span -- meaning the point, or one of the points,
    /// where the error occurred?
    is_primary: bool,
    /// Source text from the start of line_start to the end of line_end.
    text: Vec<DiagnosticSpanLine>,
    /// Label that should be placed at this location (if any)
    label: Option<String>,
    /// If we are suggesting a replacement, this will contain text
    /// that should be sliced in atop this span. You may prefer to
    /// load the fully rendered version from the parent `Diagnostic`,
    /// however.
    suggested_replacement: Option<String>,
    /// Macro invocations that created the code at this span, if any.
    expansion: Option<Box<DiagnosticSpanMacroExpansion>>,
}

#[derive(RustcEncodable)]
struct DiagnosticSpanLine {
    text: String,

    /// 1-based, character offset in self.text.
    highlight_start: usize,

    highlight_end: usize,
}

#[derive(RustcEncodable)]
struct DiagnosticSpanMacroExpansion {
    /// span where macro was applied to generate this code; note that
    /// this may itself derive from a macro (if
    /// `span.expansion.is_some()`)
    span: DiagnosticSpan,

    /// name of macro that was applied (e.g., "foo!" or "#[derive(Eq)]")
    macro_decl_name: String,

    /// span where macro was defined (if known)
    def_site_span: Option<DiagnosticSpan>,
}

#[derive(RustcEncodable)]
struct DiagnosticCode {
    /// The code itself.
    code: String,
    /// An explanation for the code.
    explanation: Option<&'static str>,
}

impl<'a> Diagnostic<'a> {
    fn from_diagnostic_builder<'c>(db: &'c DiagnosticBuilder,
                                   je: &JsonEmitter)
                                   -> Diagnostic<'c> {
        Diagnostic {
            message: &db.message,
            code: DiagnosticCode::map_opt_string(db.code.clone(), je),
            level: db.level.to_str(),
            spans: DiagnosticSpan::from_multispan(&db.span, je),
            children: db.children.iter().map(|c| {
                Diagnostic::from_sub_diagnostic(c, je)
            }).collect(),
            rendered: None,
        }
    }

    fn from_sub_diagnostic<'c>(db: &'c SubDiagnostic, je: &JsonEmitter) -> Diagnostic<'c> {
        Diagnostic {
            message: &db.message,
            code: None,
            level: db.level.to_str(),
            spans: db.render_span.as_ref()
                     .map(|sp| DiagnosticSpan::from_render_span(sp, je))
                     .unwrap_or_else(|| DiagnosticSpan::from_multispan(&db.span, je)),
            children: vec![],
            rendered: db.render_span.as_ref()
                                    .and_then(|rsp| je.render(rsp)),
        }
    }
}

impl DiagnosticSpan {
    fn from_span_label(span: SpanLabel,
                       suggestion: Option<&String>,
                       je: &JsonEmitter)
                       -> DiagnosticSpan {
        Self::from_span_etc(span.span,
                            span.is_primary,
                            span.label,
                            suggestion,
                            je)
    }

    fn from_span_etc(span: Span,
                     is_primary: bool,
                     label: Option<String>,
                     suggestion: Option<&String>,
                     je: &JsonEmitter)
                     -> DiagnosticSpan {
        // obtain the full backtrace from the `macro_backtrace`
        // helper; in some ways, it'd be better to expand the
        // backtrace ourselves, but the `macro_backtrace` helper makes
        // some decision, such as dropping some frames, and I don't
        // want to duplicate that logic here.
        let backtrace = je.cm.macro_backtrace(span).into_iter();
        DiagnosticSpan::from_span_full(span,
                                       is_primary,
                                       label,
                                       suggestion,
                                       backtrace,
                                       je)
    }

    fn from_span_full(span: Span,
                      is_primary: bool,
                      label: Option<String>,
                      suggestion: Option<&String>,
                      mut backtrace: vec::IntoIter<MacroBacktrace>,
                      je: &JsonEmitter)
                      -> DiagnosticSpan {
        let start = je.cm.lookup_char_pos(span.lo);
        let end = je.cm.lookup_char_pos(span.hi);
        let backtrace_step = backtrace.next().map(|bt| {
            let call_site =
                Self::from_span_full(bt.call_site,
                                     false,
                                     None,
                                     None,
                                     backtrace,
                                     je);
            let def_site_span = bt.def_site_span.map(|sp| {
                Self::from_span_full(sp,
                                     false,
                                     None,
                                     None,
                                     vec![].into_iter(),
                                     je)
            });
            Box::new(DiagnosticSpanMacroExpansion {
                span: call_site,
                macro_decl_name: bt.macro_decl_name,
                def_site_span: def_site_span,
            })
        });
        DiagnosticSpan {
            file_name: start.file.name.clone(),
            byte_start: span.lo.0,
            byte_end: span.hi.0,
            line_start: start.line,
            line_end: end.line,
            column_start: start.col.0 + 1,
            column_end: end.col.0 + 1,
            is_primary: is_primary,
            text: DiagnosticSpanLine::from_span(span, je),
            suggested_replacement: suggestion.cloned(),
            expansion: backtrace_step,
            label: label,
        }
    }

    fn from_multispan(msp: &MultiSpan, je: &JsonEmitter) -> Vec<DiagnosticSpan> {
        msp.span_labels()
           .into_iter()
           .map(|span_str| Self::from_span_label(span_str, None, je))
           .collect()
    }

    fn from_suggestion(suggestion: &CodeSuggestion, je: &JsonEmitter)
                       -> Vec<DiagnosticSpan> {
        assert_eq!(suggestion.msp.span_labels().len(), suggestion.substitutes.len());
        suggestion.msp.span_labels()
                      .into_iter()
                      .zip(&suggestion.substitutes)
                      .map(|(span_label, suggestion)| {
                          DiagnosticSpan::from_span_label(span_label,
                                                          Some(suggestion),
                                                          je)
                      })
                      .collect()
    }

    fn from_render_span(rsp: &RenderSpan, je: &JsonEmitter) -> Vec<DiagnosticSpan> {
        match *rsp {
            RenderSpan::FullSpan(ref msp) =>
                DiagnosticSpan::from_multispan(msp, je),
            RenderSpan::Suggestion(ref suggestion) =>
                DiagnosticSpan::from_suggestion(suggestion, je),
        }
    }
}

impl DiagnosticSpanLine {
    fn line_from_filemap(fm: &syntax_pos::FileMap,
                         index: usize,
                         h_start: usize,
                         h_end: usize)
                         -> DiagnosticSpanLine {
        DiagnosticSpanLine {
            text: fm.get_line(index).unwrap().to_owned(),
            highlight_start: h_start,
            highlight_end: h_end,
        }
    }

    /// Create a list of DiagnosticSpanLines from span - each line with any part
    /// of `span` gets a DiagnosticSpanLine, with the highlight indicating the
    /// `span` within the line.
    fn from_span(span: Span, je: &JsonEmitter) -> Vec<DiagnosticSpanLine> {
        je.cm.span_to_lines(span)
             .map(|lines| {
                 let fm = &*lines.file;
                 lines.lines
                      .iter()
                      .map(|line| {
                          DiagnosticSpanLine::line_from_filemap(fm,
                                                                line.line_index,
                                                                line.start_col.0 + 1,
                                                                line.end_col.0 + 1)
                      })
                     .collect()
             })
            .unwrap_or(vec![])
    }
}

impl DiagnosticCode {
    fn map_opt_string(s: Option<String>, je: &JsonEmitter) -> Option<DiagnosticCode> {
        s.map(|s| {

            let explanation = je.registry
                                .as_ref()
                                .and_then(|registry| registry.find_description(&s));

            DiagnosticCode {
                code: s,
                explanation: explanation,
            }
        })
    }
}

impl JsonEmitter {
    fn render(&self, render_span: &RenderSpan) -> Option<String> {
        use std::borrow::Borrow;

        match *render_span {
            RenderSpan::FullSpan(_) => {
                None
            }
            RenderSpan::Suggestion(ref suggestion) => {
                Some(suggestion.splice_lines(self.cm.borrow()))
            }
        }
    }
}

