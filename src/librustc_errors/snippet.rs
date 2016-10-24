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

use syntax_pos::{Span, FileMap};
use CodeMapper;
use std::rc::Rc;
use Level;

#[derive(Clone)]
pub struct SnippetData {
    codemap: Rc<CodeMapper>,
    files: Vec<FileInfo>,
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
}

#[derive(Clone, Debug, PartialOrd, Ord, PartialEq, Eq)]
pub struct Line {
    pub line_index: usize,
    pub annotations: Vec<Annotation>,
}

#[derive(Clone, Debug, PartialOrd, Ord, PartialEq, Eq)]
pub enum AnnotationType {
    /// Annotation under a single line of code
    Singleline,

    /// Annotation under the first character of a multiline span
    Minimized,

    /// Annotation enclosing the first and last character of a multiline span
    Multiline {
        depth: usize,
        line_start: usize,
        line_end: usize,
    },

    // The Multiline type above is replaced with the following three in order
    // to reuse the current label drawing code.
    //
    // Each of these corresponds to one part of the following diagram:
    //
    //     x |   foo(1 + bar(x,
    //       |  _________^ starting here...           < MultilineStart
    //     x | |             y),                      < MultilineLine
    //       | |______________^ ...ending here: label < MultilineEnd
    //     x |       z);
    /// Annotation marking the first character of a fully shown multiline span
    MultilineStart(usize),
    /// Annotation marking the last character of a fully shown multiline span
    MultilineEnd(usize),
    /// Line at the left enclosing the lines of a fully shown multiline span
    MultilineLine(usize),
}

impl AnnotationType {
    pub fn depth(&self) -> usize {
        match self {
            &AnnotationType::Multiline {depth, ..} |
                &AnnotationType::MultilineStart(depth) |
                &AnnotationType::MultilineLine(depth) |
                &AnnotationType::MultilineEnd(depth) => depth,
            _ => 0,
        }
    }

    pub fn increase_depth(&mut self) {
        if let AnnotationType::Multiline {ref mut depth, ..} = *self {
            *depth += 1;
        }
    }
}

#[derive(Clone, Debug, PartialOrd, Ord, PartialEq, Eq)]
pub struct Annotation {
    /// Start column, 0-based indexing -- counting *characters*, not
    /// utf-8 bytes. Note that it is important that this field goes
    /// first, so that when we sort, we sort orderings by start
    /// column.
    pub start_col: usize,

    /// End column within the line (exclusive)
    pub end_col: usize,

    /// Is this annotation derived from primary span
    pub is_primary: bool,

    /// Optional label to display adjacent to the annotation.
    pub label: Option<String>,

    /// Is this a single line, multiline or multiline span minimized down to a
    /// smaller span.
    pub annotation_type: AnnotationType,
}

impl Annotation {
    pub fn is_minimized(&self) -> bool {
        match self.annotation_type {
            AnnotationType::Minimized => true,
            _ => false,
        }
    }

    pub fn is_multiline(&self) -> bool {
        match self.annotation_type {
            AnnotationType::Multiline {..} |
                AnnotationType::MultilineStart(_) |
                AnnotationType::MultilineLine(_) |
                AnnotationType::MultilineEnd(_) => true,
            _ => false,
        }
    }

    pub fn as_start(&self) -> Annotation {
        let mut a = self.clone();
        a.annotation_type = AnnotationType::MultilineStart(self.annotation_type.depth());
        a.end_col = a.start_col + 1;
        a.label = Some("starting here...".to_owned());
        a
    }

    pub fn as_end(&self) -> Annotation {
        let mut a = self.clone();
        a.annotation_type = AnnotationType::MultilineEnd(self.annotation_type.depth());
        a.start_col = a.end_col - 1;
        a.label = match a.label {
            Some(l) => Some(format!("...ending here: {}", l)),
            None => Some("..ending here".to_owned()),
        };
        a
    }

    pub fn as_line(&self) -> Annotation {
        let mut a = self.clone();
        a.annotation_type = AnnotationType::MultilineLine(self.annotation_type.depth());
        a.label = None;
        a
    }
}

#[derive(Debug)]
pub struct StyledString {
    pub text: String,
    pub style: Style,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Style {
    HeaderMsg,
    FileNameStyle,
    LineAndColumn,
    LineNumber,
    Quotation,
    UnderlinePrimary,
    UnderlineSecondary,
    LabelPrimary,
    LabelSecondary,
    OldSchoolNoteText,
    OldSchoolNote,
    NoStyle,
    ErrorCode,
    Level(Level),
}
