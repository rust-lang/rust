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

    /// Is this a large span minimized down to a smaller span
    pub is_minimized: bool,

    /// Optional label to display adjacent to the annotation.
    pub label: Option<String>,
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
