// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Code for annotating snippets.

use syntax_pos::{Span, FileMap, CharPos, LineInfo};
use check_old_skool;
use CodeMapper;
use std::cmp;
use std::rc::Rc;
use std::mem;

#[derive(Clone)]
pub enum FormatMode {
    NewErrorFormat,
    OriginalErrorFormat,
    EnvironmentSelected
}

#[derive(Clone)]
pub struct SnippetData {
    codemap: Rc<CodeMapper>,
    files: Vec<FileInfo>,
    format_mode: FormatMode,
}

#[derive(Clone)]
pub struct FileInfo {
    file: Rc<FileMap>,

    /// The "primary file", if any, gets a `-->` marker instead of
    /// `>>>`, and has a line-number/column printed and not just a
    /// filename.  It appears first in the listing. It is known to
    /// contain at least one primary span, though primary spans (which
    /// are designated with `^^^`) may also occur in other files.
    primary_span: Option<Span>,

    lines: Vec<Line>,

    /// The type of error format to render.  We keep it here so that
    /// it's easy to configure for both tests and regular usage
    format_mode: FormatMode,
}

#[derive(Clone, Debug)]
struct Line {
    line_index: usize,
    annotations: Vec<Annotation>,
}

#[derive(Clone, Debug, PartialOrd, Ord, PartialEq, Eq)]
struct Annotation {
    /// Start column, 0-based indexing -- counting *characters*, not
    /// utf-8 bytes. Note that it is important that this field goes
    /// first, so that when we sort, we sort orderings by start
    /// column.
    start_col: usize,

    /// End column within the line (exclusive)
    end_col: usize,

    /// Is this annotation derived from primary span
    is_primary: bool,

    /// Is this a large span minimized down to a smaller span
    is_minimized: bool,

    /// Optional label to display adjacent to the annotation.
    label: Option<String>,
}

#[derive(Debug)]
pub struct RenderedLine {
    pub text: Vec<StyledString>,
    pub kind: RenderedLineKind,
}

#[derive(Debug)]
pub struct StyledString {
    pub text: String,
    pub style: Style,
}

#[derive(Debug)]
pub struct StyledBuffer {
    text: Vec<Vec<char>>,
    styles: Vec<Vec<Style>>
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Style {
    FileNameStyle,
    LineAndColumn,
    LineNumber,
    Quotation,
    UnderlinePrimary,
    UnderlineSecondary,
    LabelPrimary,
    LabelSecondary,
    OldSkoolNoteText,
    OldSkoolNote,
    NoStyle,
}

#[derive(Debug, Clone)]
pub enum RenderedLineKind {
    PrimaryFileName,
    OtherFileName,
    SourceText {
        file: Rc<FileMap>,
        line_index: usize,
    },
    Annotations,
    Elision,
}

impl SnippetData {
    pub fn new(codemap: Rc<CodeMapper>,
               primary_span: Option<Span>,
               format_mode: FormatMode) // (*)
               -> Self {
        // (*) The primary span indicates the file that must appear
        // first, and which will have a line number etc in its
        // name. Outside of tests, this is always `Some`, but for many
        // tests it's not relevant to test this portion of the logic,
        // and it's tedious to pick a primary span (read: tedious to
        // port older tests that predate the existence of a primary
        // span).

        debug!("SnippetData::new(primary_span={:?})", primary_span);

        let mut data = SnippetData {
            codemap: codemap.clone(),
            files: vec![],
            format_mode: format_mode.clone()
        };
        if let Some(primary_span) = primary_span {
            let lo = codemap.lookup_char_pos(primary_span.lo);
            data.files.push(
                FileInfo {
                    file: lo.file,
                    primary_span: Some(primary_span),
                    lines: vec![],
                    format_mode: format_mode.clone(),
                });
        }
        data
    }

    pub fn push(&mut self, span: Span, is_primary: bool, label: Option<String>) {
        debug!("SnippetData::push(span={:?}, is_primary={}, label={:?})",
               span, is_primary, label);

        let file_lines = match self.codemap.span_to_lines(span) {
            Ok(file_lines) => file_lines,
            Err(_) => {
                // ignore unprintable spans completely.
                return;
            }
        };

        self.file(&file_lines.file)
            .push_lines(&file_lines.lines, is_primary, label);
    }

    fn file(&mut self, file_map: &Rc<FileMap>) -> &mut FileInfo {
        let index = self.files.iter().position(|f| f.file.name == file_map.name);
        if let Some(index) = index {
            return &mut self.files[index];
        }

        self.files.push(
            FileInfo {
                file: file_map.clone(),
                lines: vec![],
                primary_span: None,
                format_mode: self.format_mode.clone()
            });
        self.files.last_mut().unwrap()
    }

    pub fn render_lines(&self) -> Vec<RenderedLine> {
        debug!("SnippetData::render_lines()");

        let mut rendered_lines: Vec<_> =
            self.files.iter()
                      .flat_map(|f| f.render_file_lines(&self.codemap))
                      .collect();
        prepend_prefixes(&mut rendered_lines, &self.format_mode);
        trim_lines(&mut rendered_lines);
        rendered_lines
    }
}

pub trait StringSource {
    fn make_string(self) -> String;
}

impl StringSource for String {
    fn make_string(self) -> String {
        self
    }
}

impl StringSource for Vec<char> {
    fn make_string(self) -> String {
        self.into_iter().collect()
    }
}

impl<S> From<(S, Style, RenderedLineKind)> for RenderedLine
    where S: StringSource
{
    fn from((text, style, kind): (S, Style, RenderedLineKind)) -> Self {
        RenderedLine {
            text: vec![StyledString {
                text: text.make_string(),
                style: style,
            }],
            kind: kind,
        }
    }
}

impl<S1,S2> From<(S1, Style, S2, Style, RenderedLineKind)> for RenderedLine
    where S1: StringSource, S2: StringSource
{
    fn from(tuple: (S1, Style, S2, Style, RenderedLineKind)) -> Self {
        let (text1, style1, text2, style2, kind) = tuple;
        RenderedLine {
            text: vec![
                StyledString {
                    text: text1.make_string(),
                    style: style1,
                },
                StyledString {
                    text: text2.make_string(),
                    style: style2,
                }
            ],
            kind: kind,
        }
    }
}

impl RenderedLine {
    fn trim_last(&mut self) {
        if let Some(last_text) = self.text.last_mut() {
            let len = last_text.text.trim_right().len();
            last_text.text.truncate(len);
        }
    }
}

impl RenderedLineKind {
    fn prefix(&self) -> StyledString {
        match *self {
            RenderedLineKind::SourceText { file: _, line_index } =>
                StyledString {
                    text: format!("{}", line_index + 1),
                    style: Style::LineNumber,
                },
            RenderedLineKind::Elision =>
                StyledString {
                    text: String::from("..."),
                    style: Style::LineNumber,
                },
            RenderedLineKind::PrimaryFileName |
            RenderedLineKind::OtherFileName |
            RenderedLineKind::Annotations =>
                StyledString {
                    text: String::from(""),
                    style: Style::LineNumber,
                },
        }
    }
}

impl StyledBuffer {
    fn new() -> StyledBuffer {
        StyledBuffer { text: vec![], styles: vec![] }
    }

    fn render(&self, source_kind: RenderedLineKind) -> Vec<RenderedLine> {
        let mut output: Vec<RenderedLine> = vec![];
        let mut styled_vec: Vec<StyledString> = vec![];

        for (row, row_style) in self.text.iter().zip(&self.styles) {
            let mut current_style = Style::NoStyle;
            let mut current_text = String::new();

            for (&c, &s) in row.iter().zip(row_style) {
                if s != current_style {
                    if !current_text.is_empty() {
                        styled_vec.push(StyledString { text: current_text, style: current_style });
                    }
                    current_style = s;
                    current_text = String::new();
                }
                current_text.push(c);
            }
            if !current_text.is_empty() {
                styled_vec.push(StyledString { text: current_text, style: current_style });
            }

            if output.is_empty() {
                //We know our first output line is source and the rest are highlights and labels
                output.push(RenderedLine { text: styled_vec, kind: source_kind.clone() });
            } else {
                output.push(RenderedLine { text: styled_vec, kind: RenderedLineKind::Annotations });
            }
            styled_vec = vec![];
        }

        output
    }

    fn putc(&mut self, line: usize, col: usize, chr: char, style: Style) {
        while line >= self.text.len() {
            self.text.push(vec![]);
            self.styles.push(vec![]);
        }

        if col < self.text[line].len() {
            self.text[line][col] = chr;
            self.styles[line][col] = style;
        } else {
            let mut i = self.text[line].len();
            while i < col {
                let s = match self.text[0].get(i) {
                    Some(&'\t') => '\t',
                    _ => ' '
                };
                self.text[line].push(s);
                self.styles[line].push(Style::NoStyle);
                i += 1;
            }
            self.text[line].push(chr);
            self.styles[line].push(style);
        }
    }

    fn puts(&mut self, line: usize, col: usize, string: &str, style: Style) {
        let mut n = col;
        for c in string.chars() {
            self.putc(line, n, c, style);
            n += 1;
        }
    }

    fn set_style(&mut self, line: usize, col: usize, style: Style) {
        if self.styles.len() > line && self.styles[line].len() > col {
            self.styles[line][col] = style;
        }
    }

    fn append(&mut self, line: usize, string: &str, style: Style) {
        if line >= self.text.len() {
            self.puts(line, 0, string, style);
        } else {
            let col = self.text[line].len();
            self.puts(line, col, string, style);
        }
    }
}

impl FileInfo {
    fn push_lines(&mut self,
                  lines: &[LineInfo],
                  is_primary: bool,
                  label: Option<String>) {
        assert!(lines.len() > 0);

        // If a span covers multiple lines, we reduce it to a single
        // point at the start of the span. This means that instead
        // of producing output like this:
        //
        // ```
        // --> foo.rs:2:1
        // 2   |> fn conflicting_items<'grammar>(state: &LR0State<'grammar>)
        //     |> ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        // 3   |>                               -> Set<LR0Item<'grammar>>
        //     |> ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        // (and so on)
        // ```
        //
        // we produce:
        //
        // ```
        // --> foo.rs:2:1
        // 2   |> fn conflicting_items<'grammar>(state: &LR0State<'grammar>)
        //        ^
        // ```
        //
        // Basically, although this loses information, multi-line spans just
        // never look good.

        let (line, start_col, mut end_col, is_minimized) = if lines.len() == 1 {
            (lines[0].line_index, lines[0].start_col, lines[0].end_col, false)
        } else {
            (lines[0].line_index, lines[0].start_col, CharPos(lines[0].start_col.0 + 1), true)
        };

        // Watch out for "empty spans". If we get a span like 6..6, we
        // want to just display a `^` at 6, so convert that to
        // 6..7. This is degenerate input, but it's best to degrade
        // gracefully -- and the parser likes to suply a span like
        // that for EOF, in particular.
        if start_col == end_col {
            end_col.0 += 1;
        }

        let index = self.ensure_source_line(line);
        self.lines[index].push_annotation(start_col,
                                          end_col,
                                          is_primary,
                                          is_minimized,
                                          label);
    }

    /// Ensure that we have a `Line` struct corresponding to
    /// `line_index` in the file. If we already have some other lines,
    /// then this will add the intervening lines to ensure that we
    /// have a complete snippet. (Note that when we finally display,
    /// some of those lines may be elided.)
    fn ensure_source_line(&mut self, line_index: usize) -> usize {
        if self.lines.is_empty() {
            self.lines.push(Line::new(line_index));
            return 0;
        }

        // Find the range of lines we have thus far.
        let first_line_index = self.lines.first().unwrap().line_index;
        let last_line_index = self.lines.last().unwrap().line_index;
        assert!(first_line_index <= last_line_index);

        // If the new line is lower than all the lines we have thus
        // far, then insert the new line and any intervening lines at
        // the front. In a silly attempt at micro-optimization, we
        // don't just call `insert` repeatedly, but instead make a new
        // (empty) vector, pushing the new lines onto it, and then
        // appending the old vector.
        if line_index < first_line_index {
            let lines = mem::replace(&mut self.lines, vec![]);
            self.lines.extend(
                (line_index .. first_line_index)
                    .map(|line| Line::new(line))
                    .chain(lines));
            return 0;
        }

        // If the new line comes after the ones we have so far, insert
        // lines for it.
        if line_index > last_line_index {
            self.lines.extend(
                (last_line_index+1 .. line_index+1)
                    .map(|line| Line::new(line)));
            return self.lines.len() - 1;
        }

        // Otherwise it should already exist.
        return line_index - first_line_index;
    }

    fn render_file_lines(&self, codemap: &Rc<CodeMapper>) -> Vec<RenderedLine> {
        let old_school = match self.format_mode {
            FormatMode::OriginalErrorFormat => true,
            FormatMode::NewErrorFormat => false,
            FormatMode::EnvironmentSelected => check_old_skool()
        };

        // As a first step, we elide any instance of more than one
        // continuous unannotated line.

        let mut lines_iter = self.lines.iter();
        let mut output = vec![];

        // First insert the name of the file.
        if !old_school {
            match self.primary_span {
                Some(span) => {
                    let lo = codemap.lookup_char_pos(span.lo);
                    output.push(RenderedLine {
                        text: vec![StyledString {
                            text: lo.file.name.clone(),
                            style: Style::FileNameStyle,
                        }, StyledString {
                            text: format!(":{}:{}", lo.line, lo.col.0 + 1),
                            style: Style::LineAndColumn,
                        }],
                        kind: RenderedLineKind::PrimaryFileName,
                    });
                    output.push(RenderedLine {
                        text: vec![StyledString {
                            text: "".to_string(),
                            style: Style::FileNameStyle,
                        }],
                        kind: RenderedLineKind::Annotations,
                    });
                }
                None => {
                    output.push(RenderedLine {
                        text: vec![StyledString {
                            text: self.file.name.clone(),
                            style: Style::FileNameStyle,
                        }],
                        kind: RenderedLineKind::OtherFileName,
                    });
                    output.push(RenderedLine {
                        text: vec![StyledString {
                            text: "".to_string(),
                            style: Style::FileNameStyle,
                        }],
                        kind: RenderedLineKind::Annotations,
                    });
                }
            }
        }

        let mut next_line = lines_iter.next();
        while next_line.is_some() {
            // Consume lines with annotations.
            while let Some(line) = next_line {
                if line.annotations.is_empty() { break; }

                let mut rendered_lines = self.render_line(line);
                assert!(!rendered_lines.is_empty());
                if old_school {
                    match self.primary_span {
                        Some(span) => {
                            let lo = codemap.lookup_char_pos(span.lo);
                            let hi = codemap.lookup_char_pos(span.hi);
                            //Before each secondary line in old skool-mode, print the label
                            //as an old-style note
                            if !line.annotations[0].is_primary {
                                if let Some(ann) = line.annotations[0].label.clone() {
                                    output.push(RenderedLine {
                                        text: vec![StyledString {
                                            text: lo.file.name.clone(),
                                            style: Style::FileNameStyle,
                                        }, StyledString {
                                            text: format!(":{}:{}: {}:{} ", lo.line, lo.col.0 + 1,
                                                hi.line, hi.col.0+1),
                                            style: Style::LineAndColumn,
                                        }, StyledString {
                                            text: format!("note: "),
                                            style: Style::OldSkoolNote,
                                        }, StyledString {
                                            text: format!("{}", ann),
                                            style: Style::OldSkoolNoteText,
                                        }],
                                        kind: RenderedLineKind::Annotations,
                                    });
                                }
                            }
                            rendered_lines[0].text.insert(0, StyledString {
                                text: format!(":{} ", lo.line),
                                style: Style::LineAndColumn,
                            });
                            rendered_lines[0].text.insert(0, StyledString {
                                text: lo.file.name.clone(),
                                style: Style::FileNameStyle,
                            });
                            let gap_amount =
                                rendered_lines[0].text[0].text.len() +
                                rendered_lines[0].text[1].text.len();
                            assert!(rendered_lines.len() >= 2,
                                    "no annotations resulted from: {:?}",
                                    line);
                            for i in 1..rendered_lines.len() {
                                rendered_lines[i].text.insert(0, StyledString {
                                    text: vec![" "; gap_amount].join(""),
                                    style: Style::NoStyle
                                });
                            }
                        }
                        _ =>()
                    }
                }
                output.append(&mut rendered_lines);
                next_line = lines_iter.next();
            }

            // Emit lines without annotations, but only if they are
            // followed by a line with an annotation.
            let unannotated_line = next_line;
            let mut unannotated_lines = 0;
            while let Some(line) = next_line {
                if !line.annotations.is_empty() { break; }
                unannotated_lines += 1;
                next_line = lines_iter.next();
            }
            if unannotated_lines > 1 {
                output.push(RenderedLine::from((String::new(),
                                                Style::NoStyle,
                                                RenderedLineKind::Elision)));
            } else if let Some(line) = unannotated_line {
                output.append(&mut self.render_line(line));
            }
        }

        output
    }

    fn render_line(&self, line: &Line) -> Vec<RenderedLine> {
        let old_school = match self.format_mode {
            FormatMode::OriginalErrorFormat => true,
            FormatMode::NewErrorFormat => false,
            FormatMode::EnvironmentSelected => check_old_skool()
        };

        let source_string = self.file.get_line(line.line_index)
                                     .unwrap_or("");
        let source_kind = RenderedLineKind::SourceText {
            file: self.file.clone(),
            line_index: line.line_index,
        };

        let mut styled_buffer = StyledBuffer::new();

        // First create the source line we will highlight.
        styled_buffer.append(0, &source_string, Style::Quotation);

        if line.annotations.is_empty() {
            return styled_buffer.render(source_kind);
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

        //let mut highlight_line = Self::whitespace(&source_string);

        // Sort the annotations by (start, end col)
        let mut annotations = line.annotations.clone();
        annotations.sort();

        // Next, create the highlight line.
        for annotation in &annotations {
            if old_school {
                for p in annotation.start_col .. annotation.end_col {
                    if p == annotation.start_col {
                        styled_buffer.putc(1, p, '^',
                            if annotation.is_primary {
                                Style::UnderlinePrimary
                            } else {
                                Style::OldSkoolNote
                            });
                    }
                    else {
                        styled_buffer.putc(1, p, '~',
                            if annotation.is_primary {
                                Style::UnderlinePrimary
                            } else {
                                Style::OldSkoolNote
                            });
                    }
                }
            }
            else {
                for p in annotation.start_col .. annotation.end_col {
                    if annotation.is_primary {
                        styled_buffer.putc(1, p, '^', Style::UnderlinePrimary);
                        if !annotation.is_minimized {
                            styled_buffer.set_style(0, p, Style::UnderlinePrimary);
                        }
                    } else {
                        styled_buffer.putc(1, p, '-', Style::UnderlineSecondary);
                        if !annotation.is_minimized {
                            styled_buffer.set_style(0, p, Style::UnderlineSecondary);
                        }
                    }
                }
            }
        }

        // Now we are going to write labels in. To start, we'll exclude
        // the annotations with no labels.
        let (labeled_annotations, unlabeled_annotations): (Vec<_>, _) =
            annotations.into_iter()
                       .partition(|a| a.label.is_some());

        // If there are no annotations that need text, we're done.
        if labeled_annotations.is_empty() {
            return styled_buffer.render(source_kind);
        }
        if old_school {
            return styled_buffer.render(source_kind);
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
                           .all(|a| !overlaps(a, last))
                {
                    // append the label afterwards; we keep it in a separate
                    // string
                    let highlight_label: String = format!(" {}", last.label.as_ref().unwrap());
                    if last.is_primary {
                        styled_buffer.append(1, &highlight_label, Style::LabelPrimary);
                    } else {
                        styled_buffer.append(1, &highlight_label, Style::LabelSecondary);
                    }
                    labeled_annotations = previous;
                }
            }
        }

        // If that's the last annotation, we're done
        if labeled_annotations.is_empty() {
            return styled_buffer.render(source_kind);
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
                    styled_buffer.putc(index, annotation.start_col, '|', Style::UnderlinePrimary);
                } else {
                    styled_buffer.putc(index, annotation.start_col, '|', Style::UnderlineSecondary);
                }
            }

            if annotation.is_primary {
                styled_buffer.puts(blank_lines, annotation.start_col,
                    annotation.label.as_ref().unwrap(), Style::LabelPrimary);
            } else {
                styled_buffer.puts(blank_lines, annotation.start_col,
                    annotation.label.as_ref().unwrap(), Style::LabelSecondary);
            }
        }

        styled_buffer.render(source_kind)
    }
}

fn prepend_prefixes(rendered_lines: &mut [RenderedLine], format_mode: &FormatMode) {
    let old_school = match *format_mode {
        FormatMode::OriginalErrorFormat => true,
        FormatMode::NewErrorFormat => false,
        FormatMode::EnvironmentSelected => check_old_skool()
    };
    if old_school {
        return;
    }

    let prefixes: Vec<_> =
        rendered_lines.iter()
                      .map(|rl| rl.kind.prefix())
                      .collect();

    // find the max amount of spacing we need; add 1 to
    // p.text.len() to leave space between the prefix and the
    // source text
    let padding_len =
        prefixes.iter()
                .map(|p| if p.text.len() == 0 { 0 } else { p.text.len() + 1 })
                .max()
                .unwrap_or(0);

    // Ensure we insert at least one character of padding, so that the
    // `-->` arrows can fit etc.
    let padding_len = cmp::max(padding_len, 1);

    for (mut prefix, line) in prefixes.into_iter().zip(rendered_lines) {
        let extra_spaces = (prefix.text.len() .. padding_len).map(|_| ' ');
        prefix.text.extend(extra_spaces);
        match line.kind {
            RenderedLineKind::Elision => {
                line.text.insert(0, prefix);
            }
            RenderedLineKind::PrimaryFileName => {
                //   --> filename
                // 22 |>
                //   ^
                //   padding_len
                let dashes = (0..padding_len - 1).map(|_| ' ')
                                                 .chain(Some('-'))
                                                 .chain(Some('-'))
                                                 .chain(Some('>'))
                                                 .chain(Some(' '));
                line.text.insert(0, StyledString {text: dashes.collect(),
                                                  style: Style::LineNumber})
            }
            RenderedLineKind::OtherFileName => {
                //   ::: filename
                // 22 |>
                //   ^
                //   padding_len
                let dashes = (0..padding_len - 1).map(|_| ' ')
                                                 .chain(Some(':'))
                                                 .chain(Some(':'))
                                                 .chain(Some(':'))
                                                 .chain(Some(' '));
                line.text.insert(0, StyledString {text: dashes.collect(),
                                                  style: Style::LineNumber})
            }
            _ => {
                line.text.insert(0, prefix);
                line.text.insert(1, StyledString {text: String::from("|> "),
                                                  style: Style::LineNumber})
            }
        }
    }
}

fn trim_lines(rendered_lines: &mut [RenderedLine]) {
    for line in rendered_lines {
        while !line.text.is_empty() {
            line.trim_last();
            if line.text.last().unwrap().text.is_empty() {
                line.text.pop();
            } else {
                break;
            }
        }
    }
}

impl Line {
    fn new(line_index: usize) -> Line {
        Line {
            line_index: line_index,
            annotations: vec![]
        }
    }

    fn push_annotation(&mut self,
                       start: CharPos,
                       end: CharPos,
                       is_primary: bool,
                       is_minimized: bool,
                       label: Option<String>) {
        self.annotations.push(Annotation {
            start_col: start.0,
            end_col: end.0,
            is_primary: is_primary,
            is_minimized: is_minimized,
            label: label,
        });
    }
}

fn overlaps(a1: &Annotation,
            a2: &Annotation)
            -> bool
{
    (a2.start_col .. a2.end_col).contains(a1.start_col) ||
        (a1.start_col .. a1.end_col).contains(a2.start_col)
}
