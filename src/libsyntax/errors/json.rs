// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
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


use codemap::{self, Span, MultiSpan, CodeMap};
use diagnostics::registry::Registry;
use errors::{Level, DiagnosticBuilder, SubDiagnostic, RenderSpan, CodeSuggestion};
use errors::emitter::Emitter;

use std::rc::Rc;
use std::io::{self, Write};

use rustc_serialize::json::as_json;

pub struct JsonEmitter {
    dst: Box<Write + Send>,
    registry: Option<Registry>,
    cm: Rc<CodeMap>,
}

impl JsonEmitter {
    pub fn basic() -> JsonEmitter {
        JsonEmitter::stderr(None, Rc::new(CodeMap::new()))
    }

    pub fn stderr(registry: Option<Registry>,
                  code_map: Rc<CodeMap>) -> JsonEmitter {
        JsonEmitter {
            dst: Box::new(io::stderr()),
            registry: registry,
            cm: code_map,
        }
    }
}

impl Emitter for JsonEmitter {
    fn emit(&mut self, span: Option<&MultiSpan>, msg: &str, code: Option<&str>, level: Level) {
        let data = Diagnostic::new(span, msg, code, level, self);
        if let Err(e) = writeln!(&mut self.dst, "{}", as_json(&data)) {
            panic!("failed to print diagnostics: {:?}", e);
        }
    }

    fn custom_emit(&mut self, sp: &RenderSpan, msg: &str, level: Level) {
        let data = Diagnostic::from_render_span(sp, msg, level, self);
        if let Err(e) = writeln!(&mut self.dst, "{}", as_json(&data)) {
            panic!("failed to print diagnostics: {:?}", e);
        }
    }

    fn emit_struct(&mut self, db: &DiagnosticBuilder) {
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
    /// Assocaited diagnostic messages.
    children: Vec<Diagnostic<'a>>,
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
    /// Source text from the start of line_start to the end of line_end.
    text: Vec<DiagnosticSpanLine>,
}

#[derive(RustcEncodable)]
struct DiagnosticSpanLine {
    text: String,
    /// 1-based, character offset in self.text.
    highlight_start: usize,
    highlight_end: usize,
}

#[derive(RustcEncodable)]
struct DiagnosticCode {
    /// The code itself.
    code: String,
    /// An explanation for the code.
    explanation: Option<&'static str>,
}

impl<'a> Diagnostic<'a> {
    fn new(msp: Option<&MultiSpan>,
           msg: &'a str,
           code: Option<&str>,
           level: Level,
           je: &JsonEmitter)
           -> Diagnostic<'a> {
        Diagnostic {
            message: msg,
            code: DiagnosticCode::map_opt_string(code.map(|c| c.to_owned()), je),
            level: level.to_str(),
            spans: msp.map_or(vec![], |msp| DiagnosticSpan::from_multispan(msp, je)),
            children: vec![],
        }
    }

    fn from_render_span(span: &RenderSpan,
                        msg: &'a str,
                        level: Level,
                        je: &JsonEmitter)
                        -> Diagnostic<'a> {
        Diagnostic {
            message: msg,
            code: None,
            level: level.to_str(),
            spans: DiagnosticSpan::from_render_span(span, je),
            children: vec![],
        }
    }

    fn from_diagnostic_builder<'c>(db: &'c DiagnosticBuilder,
                                   je: &JsonEmitter)
                                   -> Diagnostic<'c> {
        Diagnostic {
            message: &db.message,
            code: DiagnosticCode::map_opt_string(db.code.clone(), je),
            level: db.level.to_str(),
            spans: db.span.as_ref().map_or(vec![], |sp| DiagnosticSpan::from_multispan(sp, je)),
            children: db.children.iter().map(|c| {
                Diagnostic::from_sub_diagnostic(c, je)
            }).collect(),
        }
    }

    fn from_sub_diagnostic<'c>(db: &'c SubDiagnostic, je: &JsonEmitter) -> Diagnostic<'c> {
        Diagnostic {
            message: &db.message,
            code: None,
            level: db.level.to_str(),
            spans: db.render_span.as_ref()
                     .map(|sp| DiagnosticSpan::from_render_span(sp, je))
                     .or_else(|| db.span.as_ref().map(|s| DiagnosticSpan::from_multispan(s, je)))
                     .unwrap_or(vec![]),
            children: vec![],
        }
    }
}

impl DiagnosticSpan {
    fn from_multispan(msp: &MultiSpan, je: &JsonEmitter) -> Vec<DiagnosticSpan> {
        msp.spans.iter().map(|span| {
            let start = je.cm.lookup_char_pos(span.lo);
            let end = je.cm.lookup_char_pos(span.hi);
            DiagnosticSpan {
                file_name: start.file.name.clone(),
                byte_start: span.lo.0,
                byte_end: span.hi.0,
                line_start: start.line,
                line_end: end.line,
                column_start: start.col.0 + 1,
                column_end: end.col.0 + 1,
                text: DiagnosticSpanLine::from_span(span, je),
            }
        }).collect()
    }

    fn from_render_span(rsp: &RenderSpan, je: &JsonEmitter) -> Vec<DiagnosticSpan> {
        match *rsp {
            RenderSpan::FullSpan(ref msp) |
            // FIXME(#30701) handle Suggestion properly
            RenderSpan::Suggestion(CodeSuggestion { ref msp, .. }) => {
                DiagnosticSpan::from_multispan(msp, je)
            }
            RenderSpan::EndSpan(ref msp) => {
                msp.spans.iter().map(|span| {
                    let end = je.cm.lookup_char_pos(span.hi);
                    DiagnosticSpan {
                        file_name: end.file.name.clone(),
                        byte_start: span.hi.0,
                        byte_end: span.hi.0,
                        line_start: end.line,
                        line_end: end.line,
                        column_start: end.col.0 + 1,
                        column_end: end.col.0 + 1,
                        text: DiagnosticSpanLine::from_span_end(span, je),
                    }
                }).collect()
            }
            RenderSpan::FileLine(ref msp) => {
                msp.spans.iter().map(|span| {
                    let start = je.cm.lookup_char_pos(span.lo);
                    let end = je.cm.lookup_char_pos(span.hi);
                    DiagnosticSpan {
                        file_name: start.file.name.clone(),
                        byte_start: span.lo.0,
                        byte_end: span.hi.0,
                        line_start: start.line,
                        line_end: end.line,
                        column_start: 0,
                        column_end: 0,
                        text: DiagnosticSpanLine::from_span(span, je),
                    }
                }).collect()
            }
        }
    }
}

macro_rules! get_lines_for_span {
    ($span: ident, $je: ident) => {
        match $je.cm.span_to_lines(*$span) {
            Ok(lines) => lines,
            Err(_) => {
                debug!("unprintable span");
                return Vec::new();
            }
        }
    }
}

impl DiagnosticSpanLine {
    fn line_from_filemap(fm: &codemap::FileMap,
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
    fn from_span(span: &Span, je: &JsonEmitter) -> Vec<DiagnosticSpanLine> {
        let lines = get_lines_for_span!(span, je);

        let mut result = Vec::new();
        let fm = &*lines.file;

        for line in &lines.lines {
            result.push(DiagnosticSpanLine::line_from_filemap(fm,
                                                              line.line_index,
                                                              line.start_col.0 + 1,
                                                              line.end_col.0 + 1));
        }

        result
    }

    /// Create a list of DiagnosticSpanLines from span - the result covers all
    /// of `span`, but the highlight is zero-length and at the end of `span`.
    fn from_span_end(span: &Span, je: &JsonEmitter) -> Vec<DiagnosticSpanLine> {
        let lines = get_lines_for_span!(span, je);

        let mut result = Vec::new();
        let fm = &*lines.file;

        for (i, line) in lines.lines.iter().enumerate() {
            // Invariant - CodeMap::span_to_lines will not return extra context
            // lines - the last line returned is the last line of `span`.
            let highlight = if i == lines.lines.len() - 1 {
                (line.end_col.0 + 1, line.end_col.0 + 1)
            } else {
                (0, 0)
            };
            result.push(DiagnosticSpanLine::line_from_filemap(fm,
                                                              line.line_index,
                                                              highlight.0,
                                                              highlight.1));
        }

        result
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
