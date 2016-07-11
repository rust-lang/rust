// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use self::Destination::*;

use syntax_pos::{COMMAND_LINE_SP, DUMMY_SP, FileMap, Span, MultiSpan, LineInfo, CharPos};
use registry;

use check_old_skool;
use {Level, RenderSpan, CodeSuggestion, DiagnosticBuilder, CodeMapper};
use RenderSpan::*;
use Level::*;
use snippet::{SnippetData, StyledString, Style, FormatMode, Annotation, Line};
use styled_buffer::StyledBuffer;

use std::{cmp, fmt};
use std::io::prelude::*;
use std::io;
use std::rc::Rc;
use term;

/// Emitter trait for emitting errors.
pub trait Emitter {
    /// Emit a structured diagnostic.
    fn emit(&mut self, db: &DiagnosticBuilder);
}

impl Emitter for EmitterWriter {
    fn emit(&mut self, db: &DiagnosticBuilder) {
        if check_old_skool() {
            self.emit_message(&FullSpan(db.span.clone()),
                            &db.message,
                            db.code.as_ref().map(|s| &**s),
                            db.level,
                            true,
                            true);
            let db_span = FullSpan(db.span.clone());

            for child in &db.children {
                let render_span = child.render_span
                                    .clone()
                                    .unwrap_or_else(
                                        || FullSpan(child.span.clone()));
                let (render_span, show_snippet) = match render_span.span().primary_span() {
                    None => (db_span.clone(), false),
                    _ => (render_span, true)
                };
                self.emit_message(&render_span,
                                    &child.message,
                                    None,
                                    child.level,
                                    false,
                                    show_snippet);
            }
        } else {
            self.emit_messages_default(db);
        }
    }
}

/// maximum number of lines we will print for each error; arbitrary.
pub const MAX_HIGHLIGHT_LINES: usize = 6;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColorConfig {
    Auto,
    Always,
    Never,
}

impl ColorConfig {
    fn use_color(&self) -> bool {
        match *self {
            ColorConfig::Always => true,
            ColorConfig::Never  => false,
            ColorConfig::Auto   => stderr_isatty(),
        }
    }
}

pub struct EmitterWriter {
    dst: Destination,
    registry: Option<registry::Registry>,
    cm: Option<Rc<CodeMapper>>,

    /// Is this the first error emitted thus far? If not, we emit a
    /// `\n` before the top-level errors.
    first: bool,

    // For now, allow an old-school mode while we transition
    format_mode: FormatMode
}

struct FileWithAnnotatedLines {
    file: Rc<FileMap>,
    lines: Vec<Line>,
}


/// Do not use this for messages that end in `\n` – use `println_maybe_styled` instead. See
/// `EmitterWriter::print_maybe_styled` for details.
macro_rules! print_maybe_styled {
    ($dst: expr, $style: expr, $($arg: tt)*) => {
        $dst.print_maybe_styled(format_args!($($arg)*), $style, false)
    }
}

macro_rules! println_maybe_styled {
    ($dst: expr, $style: expr, $($arg: tt)*) => {
        $dst.print_maybe_styled(format_args!($($arg)*), $style, true)
    }
}

impl EmitterWriter {
    pub fn stderr(color_config: ColorConfig,
                  registry: Option<registry::Registry>,
                  code_map: Option<Rc<CodeMapper>>,
                  format_mode: FormatMode)
                  -> EmitterWriter {
        if color_config.use_color() {
            let dst = Destination::from_stderr();
            EmitterWriter { dst: dst,
                            registry: registry,
                            cm: code_map,
                            first: true,
                            format_mode: format_mode.clone() }
        } else {
            EmitterWriter { dst: Raw(Box::new(io::stderr())),
                            registry: registry,
                            cm: code_map,
                            first: true,
                            format_mode: format_mode.clone() }
        }
    }

    pub fn new(dst: Box<Write + Send>,
               registry: Option<registry::Registry>,
               code_map: Option<Rc<CodeMapper>>,
               format_mode: FormatMode)
               -> EmitterWriter {
        EmitterWriter { dst: Raw(dst),
                        registry: registry,
                        cm: code_map,
                        first: true,
                        format_mode: format_mode.clone() }
    }

    fn emit_message(&mut self,
                    rsp: &RenderSpan,
                    msg: &str,
                    code: Option<&str>,
                    lvl: Level,
                    is_header: bool,
                    show_snippet: bool) {
        match self.emit_message_(rsp, msg, code, lvl, is_header, show_snippet) {
            Ok(()) => { }
            Err(e) => panic!("failed to emit error: {}", e)
        }
    }

    fn preprocess_annotations(&self, msp: &MultiSpan) -> Vec<FileWithAnnotatedLines> {
        fn add_annotation_to_file(file_vec: &mut Vec<FileWithAnnotatedLines>,
                                    file: Rc<FileMap>,
                                    line_index: usize,
                                    ann: Annotation) {

            for slot in file_vec.iter_mut() {
                // Look through each of our files for the one we're adding to
                if slot.file.name == file.name {
                    // See if we already have a line for it
                    for line_slot in &mut slot.lines {
                        if line_slot.line_index == line_index {
                            line_slot.annotations.push(ann);
                            return;
                        }
                    }
                    // We don't have a line yet, create one
                    slot.lines.push(Line {
                        line_index: line_index,
                        annotations: vec![ann],
                    });
                    slot.lines.sort();
                    return;
                }
            }
            // This is the first time we're seeing the file
            file_vec.push(FileWithAnnotatedLines {
                file: file,
                lines: vec![Line {
                                line_index: line_index,
                                annotations: vec![ann],
                            }],
            });
        }

        let mut output = vec![];

        if let Some(ref cm) = self.cm {
            for span_label in msp.span_labels() {
                let lo = cm.lookup_char_pos(span_label.span.lo);
                let hi = cm.lookup_char_pos(span_label.span.hi);

                // If the span is multi-line, simplify down to the span of one character
                let (start_col, mut end_col, is_minimized) = if lo.line != hi.line {
                    (lo.col, CharPos(lo.col.0 + 1), true)
                } else {
                    (lo.col, hi.col, false)
                };

                // Watch out for "empty spans". If we get a span like 6..6, we
                // want to just display a `^` at 6, so convert that to
                // 6..7. This is degenerate input, but it's best to degrade
                // gracefully -- and the parser likes to supply a span like
                // that for EOF, in particular.
                if start_col == end_col {
                    end_col.0 += 1;
                }

                add_annotation_to_file(&mut output,
                                        lo.file,
                                        lo.line,
                                        Annotation {
                                            start_col: lo.col.0,
                                            end_col: hi.col.0,
                                            is_primary: span_label.is_primary,
                                            is_minimized: is_minimized,
                                            label: span_label.label.clone(),
                                        });
            }
        }
        output
    }

    fn render_source_line(&self,
                          buffer: &mut StyledBuffer,
                          file: Rc<FileMap>,
                          line: &Line,
                          width_offset: usize) {
        let source_string = file.get_line(line.line_index - 1)
            .unwrap_or("");

        let line_offset = buffer.num_lines();

        // First create the source line we will highlight.
        buffer.puts(line_offset, width_offset, &source_string, Style::Quotation);
        buffer.puts(line_offset,
                    0,
                    &(line.line_index.to_string()),
                    Style::LineNumber);

        draw_col_separator(buffer, line_offset, width_offset - 2);

        if line.annotations.is_empty() {
            return;
        }

        // We want to display like this:
        //
        //      vec.push(vec.pop().unwrap());
        //      ---      ^^^               _ previous borrow ends here
        //      |        |
        //      |        error occurs here
        //      previous borrow of `vec` occurs here
        //
        // But there are some weird edge cases to be aware of:
        //
        //      vec.push(vec.pop().unwrap());
        //      --------                    - previous borrow ends here
        //      ||
        //      |this makes no sense
        //      previous borrow of `vec` occurs here
        //
        // For this reason, we group the lines into "highlight lines"
        // and "annotations lines", where the highlight lines have the `~`.

        // let mut highlight_line = Self::whitespace(&source_string);
        let old_school = check_old_skool();

        // Sort the annotations by (start, end col)
        let mut annotations = line.annotations.clone();
        annotations.sort();

        // Next, create the highlight line.
        for annotation in &annotations {
            if old_school {
                for p in annotation.start_col..annotation.end_col {
                    if p == annotation.start_col {
                        buffer.putc(line_offset + 1,
                                    width_offset + p,
                                    '^',
                                    if annotation.is_primary {
                                        Style::UnderlinePrimary
                                    } else {
                                        Style::OldSchoolNote
                                    });
                    } else {
                        buffer.putc(line_offset + 1,
                                    width_offset + p,
                                    '~',
                                    if annotation.is_primary {
                                        Style::UnderlinePrimary
                                    } else {
                                        Style::OldSchoolNote
                                    });
                    }
                }
            } else {
                for p in annotation.start_col..annotation.end_col {
                    if annotation.is_primary {
                        buffer.putc(line_offset + 1,
                                    width_offset + p,
                                    '^',
                                    Style::UnderlinePrimary);
                        if !annotation.is_minimized {
                            buffer.set_style(line_offset,
                                                width_offset + p,
                                                Style::UnderlinePrimary);
                        }
                    } else {
                        buffer.putc(line_offset + 1,
                                    width_offset + p,
                                    '-',
                                    Style::UnderlineSecondary);
                        if !annotation.is_minimized {
                            buffer.set_style(line_offset,
                                                width_offset + p,
                                                Style::UnderlineSecondary);
                        }
                    }
                }
            }
        }
        draw_col_separator(buffer, line_offset + 1, width_offset - 2);

        // Now we are going to write labels in. To start, we'll exclude
        // the annotations with no labels.
        let (labeled_annotations, unlabeled_annotations): (Vec<_>, _) = annotations.into_iter()
            .partition(|a| a.label.is_some());

        // If there are no annotations that need text, we're done.
        if labeled_annotations.is_empty() {
            return;
        }
        if old_school {
            return;
        }

        // Now add the text labels. We try, when possible, to stick the rightmost
        // annotation at the end of the highlight line:
        //
        //      vec.push(vec.pop().unwrap());
        //      ---      ---               - previous borrow ends here
        //
        // But sometimes that's not possible because one of the other
        // annotations overlaps it. For example, from the test
        // `span_overlap_label`, we have the following annotations
        // (written on distinct lines for clarity):
        //
        //      fn foo(x: u32) {
        //      --------------
        //             -
        //
        // In this case, we can't stick the rightmost-most label on
        // the highlight line, or we would get:
        //
        //      fn foo(x: u32) {
        //      -------- x_span
        //      |
        //      fn_span
        //
        // which is totally weird. Instead we want:
        //
        //      fn foo(x: u32) {
        //      --------------
        //      |      |
        //      |      x_span
        //      fn_span
        //
        // which is...less weird, at least. In fact, in general, if
        // the rightmost span overlaps with any other span, we should
        // use the "hang below" version, so we can at least make it
        // clear where the span *starts*.
        let mut labeled_annotations = &labeled_annotations[..];
        match labeled_annotations.split_last().unwrap() {
            (last, previous) => {
                if previous.iter()
                    .chain(&unlabeled_annotations)
                    .all(|a| !overlaps(a, last)) {
                    // append the label afterwards; we keep it in a separate
                    // string
                    let highlight_label: String = format!(" {}", last.label.as_ref().unwrap());
                    if last.is_primary {
                        buffer.append(line_offset + 1, &highlight_label, Style::LabelPrimary);
                    } else {
                        buffer.append(line_offset + 1, &highlight_label, Style::LabelSecondary);
                    }
                    labeled_annotations = previous;
                }
            }
        }

        // If that's the last annotation, we're done
        if labeled_annotations.is_empty() {
            return;
        }

        for (index, annotation) in labeled_annotations.iter().enumerate() {
            // Leave:
            // - 1 extra line
            // - One line for each thing that comes after
            let comes_after = labeled_annotations.len() - index - 1;
            let blank_lines = 3 + comes_after;

            // For each blank line, draw a `|` at our column. The
            // text ought to be long enough for this.
            for index in 2..blank_lines {
                if annotation.is_primary {
                    buffer.putc(line_offset + index,
                                width_offset + annotation.start_col,
                                '|',
                                Style::UnderlinePrimary);
                } else {
                    buffer.putc(line_offset + index,
                                width_offset + annotation.start_col,
                                '|',
                                Style::UnderlineSecondary);
                }
                draw_col_separator(buffer, line_offset + index, width_offset - 2);
            }

            if annotation.is_primary {
                buffer.puts(line_offset + blank_lines,
                            width_offset + annotation.start_col,
                            annotation.label.as_ref().unwrap(),
                            Style::LabelPrimary);
            } else {
                buffer.puts(line_offset + blank_lines,
                            width_offset + annotation.start_col,
                            annotation.label.as_ref().unwrap(),
                            Style::LabelSecondary);
            }
            draw_col_separator(buffer, line_offset + blank_lines, width_offset - 2);
        }
    }

    fn get_multispan_max_line_num(&mut self, msp: &MultiSpan) -> usize {
        let mut max = 0;
        if let Some(ref cm) = self.cm {
            for primary_span in msp.primary_spans() {
                let hi = cm.lookup_char_pos(primary_span.hi);
                if hi.line > max {
                    max = hi.line;
                }
            }
            for span_label in msp.span_labels() {
                let hi = cm.lookup_char_pos(span_label.span.hi);
                if hi.line > max {
                    max = hi.line;
                }
            }
        }
        max
    }

    fn get_max_line_num(&mut self, db: &DiagnosticBuilder) -> usize {
        let mut max = 0;

        let primary = self.get_multispan_max_line_num(&db.span);
        max = if primary > max { primary } else { max };

        for sub in &db.children {
            let sub_result = self.get_multispan_max_line_num(&sub.span);
            max = if sub_result > max { primary } else { max };
        }
        max
    }

    fn emit_message_default(&mut self,
                            msp: &MultiSpan,
                            msg: &str,
                            code: &Option<String>,
                            level: &Level,
                            max_line_num_len: usize,
                            is_secondary: bool)
                            -> io::Result<()> {
        let mut buffer = StyledBuffer::new();

        if msp.primary_spans().is_empty() && msp.span_labels().is_empty() && is_secondary {
            // This is a secondary message with no span info
            for i in 0..max_line_num_len {
                buffer.prepend(0, " ", Style::NoStyle);
            }
            draw_note_separator(&mut buffer, 0, max_line_num_len + 1);
            buffer.append(0, &level.to_string(), Style::HeaderMsg);
            buffer.append(0, ": ", Style::NoStyle);
            buffer.append(0, msg, Style::NoStyle);
        }
        else {
            buffer.append(0, &level.to_string(), Style::Level(level.clone()));
            match code {
                &Some(ref code) => {
                    buffer.append(0, "[", Style::Level(level.clone()));
                    buffer.append(0, &code, Style::Level(level.clone()));
                    buffer.append(0, "]", Style::Level(level.clone()));
                }
                _ => {}
            }
            buffer.append(0, ": ", Style::HeaderMsg);
            buffer.append(0, msg, Style::HeaderMsg);
        }

        // Preprocess all the annotations so that they are grouped by file and by line number
        // This helps us quickly iterate over the whole message (including secondary file spans)
        let mut annotated_files = self.preprocess_annotations(msp);

        // Make sure our primary file comes first
        let primary_lo =
            if let (Some(ref cm), Some(ref primary_span)) = (self.cm.as_ref(),
                                                             msp.primary_span().as_ref()) {
                cm.lookup_char_pos(primary_span.lo)
            } else {
                // If we don't have span information, emit and exit
                emit_to_destination(&buffer.render(), level, &mut self.dst);
                return Ok(());
            };
        if let Ok(pos) =
                annotated_files.binary_search_by(|x| x.file.name.cmp(&primary_lo.file.name)) {
            annotated_files.swap(0, pos);
        }

        // Print out the annotate source lines that correspond with the error
        for annotated_file in annotated_files {
            // print out the span location and spacer before we print the annotated source
            // to do this, we need to know if this span will be primary
            let is_primary = primary_lo.file.name == annotated_file.file.name;
            if is_primary {
                // remember where we are in the output buffer for easy reference
                let mut buffer_msg_line_offset = buffer.num_lines();

                buffer.prepend(buffer_msg_line_offset, "--> ", Style::LineNumber);
                let loc = primary_lo.clone();
                buffer.append(buffer_msg_line_offset,
                                &format!("{}:{}:{}", loc.file.name, loc.line, loc.col.0),
                                Style::LineAndColumn);
                for i in 0..max_line_num_len {
                    buffer.prepend(buffer_msg_line_offset, " ", Style::NoStyle);
                }
            } else {
                // remember where we are in the output buffer for easy reference
                let mut buffer_msg_line_offset = buffer.num_lines();

                // Add spacing line
                draw_col_separator(&mut buffer, buffer_msg_line_offset, max_line_num_len + 1);

                // Then, the secondary file indicator
                buffer.prepend(buffer_msg_line_offset + 1, "::: ", Style::LineNumber);
                buffer.append(buffer_msg_line_offset + 1,
                                &annotated_file.file.name,
                                Style::LineAndColumn);
                for i in 0..max_line_num_len {
                    buffer.prepend(buffer_msg_line_offset + 1, " ", Style::NoStyle);
                }
            }

            // Put in the spacer between the location and annotated source
            let mut buffer_msg_line_offset = buffer.num_lines();
            draw_col_separator(&mut buffer, buffer_msg_line_offset, max_line_num_len + 1);

            // Next, output the annotate source for this file
            for line_idx in 0..annotated_file.lines.len() {
                self.render_source_line(&mut buffer,
                                        annotated_file.file.clone(),
                                        &annotated_file.lines[line_idx],
                                        3 + max_line_num_len);

                // check to see if we need to print out or elide lines that come between
                // this annotated line and the next one
                if line_idx < (annotated_file.lines.len() - 1) {
                    let line_idx_delta = annotated_file.lines[line_idx + 1].line_index -
                                            annotated_file.lines[line_idx].line_index;
                    if line_idx_delta > 2 {
                        let last_buffer_line_num = buffer.num_lines();
                        buffer.puts(last_buffer_line_num, 0, "...", Style::LineNumber);
                    } else if line_idx_delta == 2 {
                        let unannotated_line = annotated_file.file
                            .get_line(annotated_file.lines[line_idx].line_index)
                            .unwrap_or("");

                        let last_buffer_line_num = buffer.num_lines();

                        buffer.puts(last_buffer_line_num,
                                    0,
                                    &(annotated_file.lines[line_idx + 1].line_index - 1)
                                        .to_string(),
                                    Style::LineNumber);
                        draw_col_separator(&mut buffer, last_buffer_line_num, 1 + max_line_num_len);
                        buffer.puts(last_buffer_line_num,
                                    3 + max_line_num_len,
                                    &unannotated_line,
                                    Style::Quotation);
                    }
                }
            }
        }

        // final step: take our styled buffer, render it, then output it
        emit_to_destination(&buffer.render(), level, &mut self.dst);

        Ok(())
    }
    fn emit_suggestion_default(&mut self,
                               suggestion: &CodeSuggestion,
                               level: &Level,
                               msg: &str,
                               max_line_num_len: usize)
                               -> io::Result<()> {
        use std::borrow::Borrow;

        let primary_span = suggestion.msp.primary_span().unwrap();
        if let Some(ref cm) = self.cm {
            let mut buffer = StyledBuffer::new();

            buffer.append(0, &level.to_string(), Style::Level(level.clone()));
            buffer.append(0, ": ", Style::HeaderMsg);
            buffer.append(0, msg, Style::HeaderMsg);

            let lines = cm.span_to_lines(primary_span).unwrap();

            assert!(!lines.lines.is_empty());

            let complete = suggestion.splice_lines(cm.borrow());
            let line_count = cmp::min(lines.lines.len(), MAX_HIGHLIGHT_LINES);
            let display_lines = &lines.lines[..line_count];

            let fm = &*lines.file;
            // Calculate the widest number to format evenly
            let max_digits = line_num_max_digits(display_lines.last().unwrap());

            // print the suggestion without any line numbers, but leave
            // space for them. This helps with lining up with previous
            // snippets from the actual error being reported.
            let mut lines = complete.lines();
            let mut row_num = 1;
            for line in lines.by_ref().take(MAX_HIGHLIGHT_LINES) {
                draw_col_separator(&mut buffer, row_num, max_line_num_len + 1);
                buffer.append(row_num, line, Style::NoStyle);
                row_num += 1;
            }

            // if we elided some lines, add an ellipsis
            if let Some(_) = lines.next() {
                buffer.append(row_num, "...", Style::NoStyle);
            }
            emit_to_destination(&buffer.render(), level, &mut self.dst);
        }
        Ok(())
    }
    fn emit_messages_default(&mut self, db: &DiagnosticBuilder) {
        let max_line_num = self.get_max_line_num(db);
        let max_line_num_len = max_line_num.to_string().len();

        match self.emit_message_default(&db.span,
                                        &db.message,
                                        &db.code,
                                        &db.level,
                                        max_line_num_len,
                                        false) {
            Ok(()) => {
                if !db.children.is_empty() {
                    let mut buffer = StyledBuffer::new();
                    draw_col_separator(&mut buffer, 0, max_line_num_len + 1);
                    emit_to_destination(&buffer.render(), &db.level, &mut self.dst);
                }
                for child in &db.children {
                    match child.render_span {
                        Some(FullSpan(ref msp)) => {
                            match self.emit_message_default(msp,
                                                            &child.message,
                                                            &None,
                                                            &child.level,
                                                            max_line_num_len,
                                                            true) {
                                Err(e) => panic!("failed to emit error: {}", e),
                                _ => ()
                            }
                        },
                        Some(Suggestion(ref cs)) => {
                            match self.emit_suggestion_default(cs,
                                                               &child.level,
                                                               &child.message,
                                                               max_line_num_len) {
                                Err(e) => panic!("failed to emit error: {}", e),
                                _ => ()
                            }
                        },
                        None => {
                            match self.emit_message_default(&child.span,
                                                            &child.message,
                                                            &None,
                                                            &child.level,
                                                            max_line_num_len,
                                                            true) {
                                Err(e) => panic!("failed to emit error: {}", e),
                                _ => ()
                            }
                        }
                    }
                }
            }
            Err(e) => panic!("failed to emit error: {}", e)
        }
        write!(&mut self.dst, "\n");
    }

    fn emit_message_(&mut self,
                     rsp: &RenderSpan,
                     msg: &str,
                     code: Option<&str>,
                     lvl: Level,
                     is_header: bool,
                     show_snippet: bool)
                     -> io::Result<()> {
        let old_school = match self.format_mode {
            FormatMode::NewErrorFormat => false,
            FormatMode::OriginalErrorFormat => true,
            FormatMode::EnvironmentSelected => check_old_skool()
        };

        if is_header {
            if self.first {
                self.first = false;
            } else {
                if !old_school {
                    write!(self.dst, "\n")?;
                }
            }
        }

        match code {
            Some(code) if self.registry.as_ref()
                                       .and_then(|registry| registry.find_description(code))
                                       .is_some() => {
                let code_with_explain = String::from("--explain ") + code;
                if old_school {
                    let loc = match rsp.span().primary_span() {
                        Some(COMMAND_LINE_SP) | Some(DUMMY_SP) => "".to_string(),
                        Some(ps) => if let Some(ref cm) = self.cm {
                            cm.span_to_string(ps)
                        } else {
                            "".to_string()
                        },
                        None => "".to_string()
                    };
                    print_diagnostic(&mut self.dst, &loc, lvl, msg, Some(code))?
                }
                else {
                    print_diagnostic(&mut self.dst, "", lvl, msg, Some(&code_with_explain))?
                }
            }
            _ => {
                if old_school {
                    let loc = match rsp.span().primary_span() {
                        Some(COMMAND_LINE_SP) | Some(DUMMY_SP) => "".to_string(),
                        Some(ps) => if let Some(ref cm) = self.cm {
                            cm.span_to_string(ps)
                        } else {
                            "".to_string()
                        },
                        None => "".to_string()
                    };
                    print_diagnostic(&mut self.dst, &loc, lvl, msg, code)?
                }
                else {
                    print_diagnostic(&mut self.dst, "", lvl, msg, code)?
                }
            }
        }

        if !show_snippet {
            return Ok(());
        }

        // Watch out for various nasty special spans; don't try to
        // print any filename or anything for those.
        match rsp.span().primary_span() {
            Some(COMMAND_LINE_SP) | Some(DUMMY_SP) => {
                return Ok(());
            }
            _ => { }
        }

        // Otherwise, print out the snippet etc as needed.
        match *rsp {
            FullSpan(ref msp) => {
                self.highlight_lines(msp, lvl)?;
                if let Some(primary_span) = msp.primary_span() {
                    self.print_macro_backtrace(primary_span)?;
                }
            }
            Suggestion(ref suggestion) => {
                self.highlight_suggestion(suggestion)?;
                if let Some(primary_span) = rsp.span().primary_span() {
                    self.print_macro_backtrace(primary_span)?;
                }
            }
        }
        if old_school {
            match code {
                Some(code) if self.registry.as_ref()
                                        .and_then(|registry| registry.find_description(code))
                                        .is_some() => {
                    let loc = match rsp.span().primary_span() {
                        Some(COMMAND_LINE_SP) | Some(DUMMY_SP) => "".to_string(),
                        Some(ps) => if let Some(ref cm) = self.cm {
                            cm.span_to_string(ps)
                        } else {
                            "".to_string()
                        },
                        None => "".to_string()
                    };
                    let msg = "run `rustc --explain ".to_string() + &code.to_string() +
                        "` to see a detailed explanation";
                    print_diagnostic(&mut self.dst, &loc, Level::Help, &msg,
                        None)?
                }
                _ => ()
            }
        }
        Ok(())
    }

    fn highlight_suggestion(&mut self, suggestion: &CodeSuggestion) -> io::Result<()>
    {
        use std::borrow::Borrow;

        let primary_span = suggestion.msp.primary_span().unwrap();
        if let Some(ref cm) = self.cm {
            let lines = cm.span_to_lines(primary_span).unwrap();

            assert!(!lines.lines.is_empty());

            let complete = suggestion.splice_lines(cm.borrow());
            let line_count = cmp::min(lines.lines.len(), MAX_HIGHLIGHT_LINES);
            let display_lines = &lines.lines[..line_count];

            let fm = &*lines.file;
            // Calculate the widest number to format evenly
            let max_digits = line_num_max_digits(display_lines.last().unwrap());

            // print the suggestion without any line numbers, but leave
            // space for them. This helps with lining up with previous
            // snippets from the actual error being reported.
            let mut lines = complete.lines();
            for line in lines.by_ref().take(MAX_HIGHLIGHT_LINES) {
                write!(&mut self.dst, "{0}:{1:2$} {3}\n",
                    fm.name, "", max_digits, line)?;
            }

            // if we elided some lines, add an ellipsis
            if let Some(_) = lines.next() {
                write!(&mut self.dst, "{0:1$} {0:2$} ...\n",
                    "", fm.name.len(), max_digits)?;
            }
        }
        Ok(())
    }

    pub fn highlight_lines(&mut self,
                       msp: &MultiSpan,
                       lvl: Level)
                       -> io::Result<()>
    {
        // Check to see if we have any lines to highlight, exit early if not
        match self.cm {
            None => return Ok(()),
            _ => ()
        }

        let old_school = match self.format_mode {
            FormatMode::NewErrorFormat => false,
            FormatMode::OriginalErrorFormat => true,
            FormatMode::EnvironmentSelected => check_old_skool()
        };

        let mut snippet_data = SnippetData::new(self.cm.as_ref().unwrap().clone(),
                                                msp.primary_span(),
                                                self.format_mode.clone());
        if old_school {
            let mut output_vec = vec![];

            for span_label in msp.span_labels() {
                let mut snippet_data = SnippetData::new(self.cm.as_ref().unwrap().clone(),
                                                        Some(span_label.span),
                                                        self.format_mode.clone());

                snippet_data.push(span_label.span,
                                  span_label.is_primary,
                                  span_label.label);
                if span_label.is_primary {
                    output_vec.insert(0, snippet_data);
                }
                else {
                    output_vec.push(snippet_data);
                }
            }

            for snippet_data in output_vec.iter() {
                /*
                let rendered_lines = snippet_data.render_lines();
                for rendered_line in &rendered_lines {
                    for styled_string in &rendered_line.text {
                        self.dst.apply_style(lvl, &rendered_line.kind, styled_string.style)?;
                        write!(&mut self.dst, "{}", styled_string.text)?;
                        self.dst.reset_attrs()?;
                    }
                    write!(&mut self.dst, "\n")?;
                }
                */
                emit_to_destination(&snippet_data.render_lines(), &lvl, &mut self.dst);
            }
        }
        else {
            for span_label in msp.span_labels() {
                snippet_data.push(span_label.span,
                                  span_label.is_primary,
                                  span_label.label);
            }
            emit_to_destination(&snippet_data.render_lines(), &lvl, &mut self.dst);
            /*
            let rendered_lines = snippet_data.render_lines();
            for rendered_line in &rendered_lines {
                for styled_string in &rendered_line.text {
                    self.dst.apply_style(lvl, &rendered_line.kind, styled_string.style)?;
                    write!(&mut self.dst, "{}", styled_string.text)?;
                    self.dst.reset_attrs()?;
                }
                write!(&mut self.dst, "\n")?;
            }
            */
        }
        Ok(())
    }

    fn print_macro_backtrace(&mut self,
                             sp: Span)
                             -> io::Result<()> {
        if let Some(ref cm) = self.cm {
            for trace in cm.macro_backtrace(sp) {
                let mut diag_string =
                    format!("in this expansion of {}", trace.macro_decl_name);
                if let Some(def_site_span) = trace.def_site_span {
                    diag_string.push_str(
                        &format!(" (defined in {})",
                            cm.span_to_filename(def_site_span)));
                }
                let snippet = cm.span_to_string(trace.call_site);
                print_diagnostic(&mut self.dst, &snippet, Note, &diag_string, None)?;
            }
        }
        Ok(())
    }
}

fn draw_col_separator(buffer: &mut StyledBuffer, line: usize, col: usize) {
    buffer.puts(line, col, "| ", Style::LineNumber);
}

fn draw_note_separator(buffer: &mut StyledBuffer, line: usize, col: usize) {
    buffer.puts(line, col, "= ", Style::LineNumber);
}

fn overlaps(a1: &Annotation, a2: &Annotation) -> bool {
    (a2.start_col..a2.end_col).contains(a1.start_col) ||
    (a1.start_col..a1.end_col).contains(a2.start_col)
}

fn emit_to_destination(rendered_buffer: &Vec<Vec<StyledString>>,
        lvl: &Level,
        dst: &mut Destination) -> io::Result<()> {
    for line in rendered_buffer {
        for part in line {
            dst.apply_style(lvl.clone(), part.style);
            write!(dst, "{}", part.text);
            dst.reset_attrs()?;
        }
        write!(dst, "\n");
    }
    Ok(())
}

fn line_num_max_digits(line: &LineInfo) -> usize {
    let mut max_line_num = line.line_index + 1;
    let mut digits = 0;
    while max_line_num > 0 {
        max_line_num /= 10;
        digits += 1;
    }
    digits
}

fn print_diagnostic(dst: &mut Destination,
                    topic: &str,
                    lvl: Level,
                    msg: &str,
                    code: Option<&str>)
                    -> io::Result<()> {
    if !topic.is_empty() {
        let old_school = check_old_skool();
        if !old_school {
            write!(dst, "{}: ", topic)?;
        }
        else {
            write!(dst, "{} ", topic)?;
        }
        dst.reset_attrs()?;
    }
    dst.start_attr(term::Attr::Bold)?;
    dst.start_attr(term::Attr::ForegroundColor(lvl.color()))?;
    write!(dst, "{}", lvl.to_string())?;
    dst.reset_attrs()?;
    write!(dst, ": ")?;
    dst.start_attr(term::Attr::Bold)?;
    write!(dst, "{}", msg)?;

    if let Some(code) = code {
        let style = term::Attr::ForegroundColor(term::color::BRIGHT_MAGENTA);
        print_maybe_styled!(dst, style, " [{}]", code.clone())?;
    }

    dst.reset_attrs()?;
    write!(dst, "\n")?;
    Ok(())
}

#[cfg(unix)]
fn stderr_isatty() -> bool {
    use libc;
    unsafe { libc::isatty(libc::STDERR_FILENO) != 0 }
}
#[cfg(windows)]
fn stderr_isatty() -> bool {
    type DWORD = u32;
    type BOOL = i32;
    type HANDLE = *mut u8;
    const STD_ERROR_HANDLE: DWORD = -12i32 as DWORD;
    extern "system" {
        fn GetStdHandle(which: DWORD) -> HANDLE;
        fn GetConsoleMode(hConsoleHandle: HANDLE,
                          lpMode: *mut DWORD) -> BOOL;
    }
    unsafe {
        let handle = GetStdHandle(STD_ERROR_HANDLE);
        let mut out = 0;
        GetConsoleMode(handle, &mut out) != 0
    }
}

pub enum Destination {
    Terminal(Box<term::StderrTerminal>),
    Raw(Box<Write + Send>),
}

impl Destination {
    fn from_stderr() -> Destination {
        match term::stderr() {
            Some(t) => Terminal(t),
            None    => Raw(Box::new(io::stderr())),
        }
    }

    fn apply_style(&mut self,
                   lvl: Level,
                   style: Style)
                   -> io::Result<()> {
        match style {
            Style::FileNameStyle | Style::LineAndColumn => {}
            Style::LineNumber => {
                try!(self.start_attr(term::Attr::Bold));
                try!(self.start_attr(term::Attr::ForegroundColor(term::color::BRIGHT_BLUE)));
            }
            Style::ErrorCode => {
                try!(self.start_attr(term::Attr::Bold));
                //try!(self.start_attr(term::Attr::ForegroundColor(term::color::BRIGHT_MAGENTA)));
            }
            Style::Quotation => {}
            Style::OldSchoolNote => {
                try!(self.start_attr(term::Attr::Bold));
                try!(self.start_attr(term::Attr::ForegroundColor(term::color::BRIGHT_GREEN)));
            }
            Style::OldSchoolNoteText | Style::HeaderMsg => {
                try!(self.start_attr(term::Attr::Bold));
            }
            Style::UnderlinePrimary | Style::LabelPrimary => {
                try!(self.start_attr(term::Attr::Bold));
                try!(self.start_attr(term::Attr::ForegroundColor(lvl.color())));
            }
            Style::UnderlineSecondary |
            Style::LabelSecondary => {
                try!(self.start_attr(term::Attr::Bold));
                try!(self.start_attr(term::Attr::ForegroundColor(term::color::BRIGHT_BLUE)));
            }
            Style::NoStyle => {}
            Style::Level(l) => {
                try!(self.start_attr(term::Attr::Bold));
                try!(self.start_attr(term::Attr::ForegroundColor(l.color())));
            }
        }
        Ok(())
    }

    fn start_attr(&mut self, attr: term::Attr) -> io::Result<()> {
        match *self {
            Terminal(ref mut t) => { t.attr(attr)?; }
            Raw(_) => { }
        }
        Ok(())
    }

    fn reset_attrs(&mut self) -> io::Result<()> {
        match *self {
            Terminal(ref mut t) => { t.reset()?; }
            Raw(_) => { }
        }
        Ok(())
    }

    fn print_maybe_styled(&mut self,
                          args: fmt::Arguments,
                          color: term::Attr,
                          print_newline_at_end: bool)
                          -> io::Result<()> {
        match *self {
            Terminal(ref mut t) => {
                t.attr(color)?;
                // If `msg` ends in a newline, we need to reset the color before
                // the newline. We're making the assumption that we end up writing
                // to a `LineBufferedWriter`, which means that emitting the reset
                // after the newline ends up buffering the reset until we print
                // another line or exit. Buffering the reset is a problem if we're
                // sharing the terminal with any other programs (e.g. other rustc
                // instances via `make -jN`).
                //
                // Note that if `msg` contains any internal newlines, this will
                // result in the `LineBufferedWriter` flushing twice instead of
                // once, which still leaves the opportunity for interleaved output
                // to be miscolored. We assume this is rare enough that we don't
                // have to worry about it.
                t.write_fmt(args)?;
                t.reset()?;
                if print_newline_at_end {
                    t.write_all(b"\n")
                } else {
                    Ok(())
                }
            }
            Raw(ref mut w) => {
                w.write_fmt(args)?;
                if print_newline_at_end {
                    w.write_all(b"\n")
                } else {
                    Ok(())
                }
            }
        }
    }
}

impl Write for Destination {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        match *self {
            Terminal(ref mut t) => t.write(bytes),
            Raw(ref mut w) => w.write(bytes),
        }
    }
    fn flush(&mut self) -> io::Result<()> {
        match *self {
            Terminal(ref mut t) => t.flush(),
            Raw(ref mut w) => w.flush(),
        }
    }
}
