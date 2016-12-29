// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! The source positions and related helper functions
//!
//! # Note
//!
//! This API is completely unstable and subject to change.

#![crate_name = "syntax_pos"]
#![unstable(feature = "rustc_private", issue = "27812")]
#![crate_type = "dylib"]
#![crate_type = "rlib"]
#![doc(html_logo_url = "https://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
      html_favicon_url = "https://doc.rust-lang.org/favicon.ico",
      html_root_url = "https://doc.rust-lang.org/nightly/")]
#![deny(warnings)]

#![feature(custom_attribute)]
#![allow(unused_attributes)]
#![feature(rustc_private)]
#![feature(staged_api)]
#![feature(specialization)]

use std::cell::{Cell, RefCell};
use std::ops::{Add, Sub};
use std::rc::Rc;
use std::cmp;

use std::fmt;

use serialize::{Encodable, Decodable, Encoder, Decoder};

extern crate serialize;
extern crate serialize as rustc_serialize; // used by deriving

pub type FileName = String;

/// Spans represent a region of code, used for error reporting. Positions in spans
/// are *absolute* positions from the beginning of the codemap, not positions
/// relative to FileMaps. Methods on the CodeMap can be used to relate spans back
/// to the original source.
/// You must be careful if the span crosses more than one file - you will not be
/// able to use many of the functions on spans in codemap and you cannot assume
/// that the length of the span = hi - lo; there may be space in the BytePos
/// range between files.
#[derive(Clone, Copy, Hash, PartialEq, Eq, Ord, PartialOrd)]
pub struct Span {
    pub lo: BytePos,
    pub hi: BytePos,
    /// Information about where the macro came from, if this piece of
    /// code was created by a macro expansion.
    pub expn_id: ExpnId
}

/// A collection of spans. Spans have two orthogonal attributes:
///
/// - they can be *primary spans*. In this case they are the locus of
///   the error, and would be rendered with `^^^`.
/// - they can have a *label*. In this case, the label is written next
///   to the mark in the snippet when we render.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct MultiSpan {
    primary_spans: Vec<Span>,
    span_labels: Vec<(Span, String)>,
}

impl Span {
    /// Returns a new span representing just the end-point of this span
    pub fn end_point(self) -> Span {
        let lo = cmp::max(self.hi.0 - 1, self.lo.0);
        Span { lo: BytePos(lo), hi: self.hi, expn_id: self.expn_id}
    }

    /// Returns `self` if `self` is not the dummy span, and `other` otherwise.
    pub fn substitute_dummy(self, other: Span) -> Span {
        if self.source_equal(&DUMMY_SP) { other } else { self }
    }

    pub fn contains(self, other: Span) -> bool {
        self.lo <= other.lo && other.hi <= self.hi
    }

    /// Return true if the spans are equal with regards to the source text.
    ///
    /// Use this instead of `==` when either span could be generated code,
    /// and you only care that they point to the same bytes of source text.
    pub fn source_equal(&self, other: &Span) -> bool {
        self.lo == other.lo && self.hi == other.hi
    }

    /// Returns `Some(span)`, where the start is trimmed by the end of `other`
    pub fn trim_start(self, other: Span) -> Option<Span> {
        if self.hi > other.hi {
            Some(Span { lo: cmp::max(self.lo, other.hi), .. self })
        } else {
            None
        }
    }
}

#[derive(Clone, Debug)]
pub struct SpanLabel {
    /// The span we are going to include in the final snippet.
    pub span: Span,

    /// Is this a primary span? This is the "locus" of the message,
    /// and is indicated with a `^^^^` underline, versus `----`.
    pub is_primary: bool,

    /// What label should we attach to this span (if any)?
    pub label: Option<String>,
}

impl serialize::UseSpecializedEncodable for Span {
    fn default_encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        s.emit_struct("Span", 2, |s| {
            s.emit_struct_field("lo", 0, |s| {
                self.lo.encode(s)
            })?;

            s.emit_struct_field("hi", 1, |s| {
                self.hi.encode(s)
            })
        })
    }
}

impl serialize::UseSpecializedDecodable for Span {
    fn default_decode<D: Decoder>(d: &mut D) -> Result<Span, D::Error> {
        d.read_struct("Span", 2, |d| {
            let lo = d.read_struct_field("lo", 0, Decodable::decode)?;
            let hi = d.read_struct_field("hi", 1, Decodable::decode)?;
            Ok(mk_sp(lo, hi))
        })
    }
}

fn default_span_debug(span: Span, f: &mut fmt::Formatter) -> fmt::Result {
    write!(f, "Span {{ lo: {:?}, hi: {:?}, expn_id: {:?} }}",
           span.lo, span.hi, span.expn_id)
}

impl fmt::Debug for Span {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        SPAN_DEBUG.with(|span_debug| span_debug.get()(*self, f))
    }
}

pub const DUMMY_SP: Span = Span { lo: BytePos(0), hi: BytePos(0), expn_id: NO_EXPANSION };

// Generic span to be used for code originating from the command line
pub const COMMAND_LINE_SP: Span = Span { lo: BytePos(0),
                                         hi: BytePos(0),
                                         expn_id: COMMAND_LINE_EXPN };

impl MultiSpan {
    pub fn new() -> MultiSpan {
        MultiSpan {
            primary_spans: vec![],
            span_labels: vec![]
        }
    }

    pub fn from_span(primary_span: Span) -> MultiSpan {
        MultiSpan {
            primary_spans: vec![primary_span],
            span_labels: vec![]
        }
    }

    pub fn from_spans(vec: Vec<Span>) -> MultiSpan {
        MultiSpan {
            primary_spans: vec,
            span_labels: vec![]
        }
    }

    pub fn push_span_label(&mut self, span: Span, label: String) {
        self.span_labels.push((span, label));
    }

    /// Selects the first primary span (if any)
    pub fn primary_span(&self) -> Option<Span> {
        self.primary_spans.first().cloned()
    }

    /// Returns all primary spans.
    pub fn primary_spans(&self) -> &[Span] {
        &self.primary_spans
    }

    /// Replaces all occurances of one Span with another. Used to move Spans in areas that don't
    /// display well (like std macros). Returns true if replacements occurred.
    pub fn replace(&mut self, before: Span, after: Span) -> bool {
        let mut replacements_occurred = false;
        for primary_span in &mut self.primary_spans {
            if *primary_span == before {
                *primary_span = after;
                replacements_occurred = true;
            }
        }
        for span_label in &mut self.span_labels {
            if span_label.0 == before {
                span_label.0 = after;
                replacements_occurred = true;
            }
        }
        replacements_occurred
    }

    /// Returns the strings to highlight. We always ensure that there
    /// is an entry for each of the primary spans -- for each primary
    /// span P, if there is at least one label with span P, we return
    /// those labels (marked as primary). But otherwise we return
    /// `SpanLabel` instances with empty labels.
    pub fn span_labels(&self) -> Vec<SpanLabel> {
        let is_primary = |span| self.primary_spans.contains(&span);
        let mut span_labels = vec![];

        for &(span, ref label) in &self.span_labels {
            span_labels.push(SpanLabel {
                span: span,
                is_primary: is_primary(span),
                label: Some(label.clone())
            });
        }

        for &span in &self.primary_spans {
            if !span_labels.iter().any(|sl| sl.span == span) {
                span_labels.push(SpanLabel {
                    span: span,
                    is_primary: true,
                    label: None
                });
            }
        }

        span_labels
    }
}

impl From<Span> for MultiSpan {
    fn from(span: Span) -> MultiSpan {
        MultiSpan::from_span(span)
    }
}

#[derive(PartialEq, Eq, Clone, Debug, Hash, RustcEncodable, RustcDecodable, Copy, Ord, PartialOrd)]
pub struct ExpnId(pub u32);

pub const NO_EXPANSION: ExpnId = ExpnId(!0);
// For code appearing from the command line
pub const COMMAND_LINE_EXPN: ExpnId = ExpnId(!1);

// For code generated by a procedural macro, without knowing which
// Used in `qquote!`
pub const PROC_EXPN: ExpnId = ExpnId(!2);

impl ExpnId {
    pub fn from_u32(id: u32) -> ExpnId {
        ExpnId(id)
    }

    pub fn into_u32(self) -> u32 {
        self.0
    }
}

/// Identifies an offset of a multi-byte character in a FileMap
#[derive(Copy, Clone, RustcEncodable, RustcDecodable, Eq, PartialEq)]
pub struct MultiByteChar {
    /// The absolute offset of the character in the CodeMap
    pub pos: BytePos,
    /// The number of bytes, >=2
    pub bytes: usize,
}

/// A single source in the CodeMap.
pub struct FileMap {
    /// The name of the file that the source came from, source that doesn't
    /// originate from files has names between angle brackets by convention,
    /// e.g. `<anon>`
    pub name: FileName,
    /// The absolute path of the file that the source came from.
    pub abs_path: Option<FileName>,
    /// The complete source code
    pub src: Option<Rc<String>>,
    /// The start position of this source in the CodeMap
    pub start_pos: BytePos,
    /// The end position of this source in the CodeMap
    pub end_pos: BytePos,
    /// Locations of lines beginnings in the source code
    pub lines: RefCell<Vec<BytePos>>,
    /// Locations of multi-byte characters in the source code
    pub multibyte_chars: RefCell<Vec<MultiByteChar>>,
}

impl Encodable for FileMap {
    fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        s.emit_struct("FileMap", 6, |s| {
            s.emit_struct_field("name", 0, |s| self.name.encode(s))?;
            s.emit_struct_field("abs_path", 1, |s| self.abs_path.encode(s))?;
            s.emit_struct_field("start_pos", 2, |s| self.start_pos.encode(s))?;
            s.emit_struct_field("end_pos", 3, |s| self.end_pos.encode(s))?;
            s.emit_struct_field("lines", 4, |s| {
                let lines = self.lines.borrow();
                // store the length
                s.emit_u32(lines.len() as u32)?;

                if !lines.is_empty() {
                    // In order to preserve some space, we exploit the fact that
                    // the lines list is sorted and individual lines are
                    // probably not that long. Because of that we can store lines
                    // as a difference list, using as little space as possible
                    // for the differences.
                    let max_line_length = if lines.len() == 1 {
                        0
                    } else {
                        lines.windows(2)
                             .map(|w| w[1] - w[0])
                             .map(|bp| bp.to_usize())
                             .max()
                             .unwrap()
                    };

                    let bytes_per_diff: u8 = match max_line_length {
                        0 ... 0xFF => 1,
                        0x100 ... 0xFFFF => 2,
                        _ => 4
                    };

                    // Encode the number of bytes used per diff.
                    bytes_per_diff.encode(s)?;

                    // Encode the first element.
                    lines[0].encode(s)?;

                    let diff_iter = (&lines[..]).windows(2)
                                                .map(|w| (w[1] - w[0]));

                    match bytes_per_diff {
                        1 => for diff in diff_iter { (diff.0 as u8).encode(s)? },
                        2 => for diff in diff_iter { (diff.0 as u16).encode(s)? },
                        4 => for diff in diff_iter { diff.0.encode(s)? },
                        _ => unreachable!()
                    }
                }

                Ok(())
            })?;
            s.emit_struct_field("multibyte_chars", 5, |s| {
                (*self.multibyte_chars.borrow()).encode(s)
            })
        })
    }
}

impl Decodable for FileMap {
    fn decode<D: Decoder>(d: &mut D) -> Result<FileMap, D::Error> {

        d.read_struct("FileMap", 6, |d| {
            let name: String = d.read_struct_field("name", 0, |d| Decodable::decode(d))?;
            let abs_path: Option<String> =
                d.read_struct_field("abs_path", 1, |d| Decodable::decode(d))?;
            let start_pos: BytePos = d.read_struct_field("start_pos", 2, |d| Decodable::decode(d))?;
            let end_pos: BytePos = d.read_struct_field("end_pos", 3, |d| Decodable::decode(d))?;
            let lines: Vec<BytePos> = d.read_struct_field("lines", 4, |d| {
                let num_lines: u32 = Decodable::decode(d)?;
                let mut lines = Vec::with_capacity(num_lines as usize);

                if num_lines > 0 {
                    // Read the number of bytes used per diff.
                    let bytes_per_diff: u8 = Decodable::decode(d)?;

                    // Read the first element.
                    let mut line_start: BytePos = Decodable::decode(d)?;
                    lines.push(line_start);

                    for _ in 1..num_lines {
                        let diff = match bytes_per_diff {
                            1 => d.read_u8()? as u32,
                            2 => d.read_u16()? as u32,
                            4 => d.read_u32()?,
                            _ => unreachable!()
                        };

                        line_start = line_start + BytePos(diff);

                        lines.push(line_start);
                    }
                }

                Ok(lines)
            })?;
            let multibyte_chars: Vec<MultiByteChar> =
                d.read_struct_field("multibyte_chars", 5, |d| Decodable::decode(d))?;
            Ok(FileMap {
                name: name,
                abs_path: abs_path,
                start_pos: start_pos,
                end_pos: end_pos,
                src: None,
                lines: RefCell::new(lines),
                multibyte_chars: RefCell::new(multibyte_chars)
            })
        })
    }
}

impl fmt::Debug for FileMap {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "FileMap({})", self.name)
    }
}

impl FileMap {
    /// EFFECT: register a start-of-line offset in the
    /// table of line-beginnings.
    /// UNCHECKED INVARIANT: these offsets must be added in the right
    /// order and must be in the right places; there is shared knowledge
    /// about what ends a line between this file and parse.rs
    /// WARNING: pos param here is the offset relative to start of CodeMap,
    /// and CodeMap will append a newline when adding a filemap without a newline at the end,
    /// so the safe way to call this is with value calculated as
    /// filemap.start_pos + newline_offset_relative_to_the_start_of_filemap.
    pub fn next_line(&self, pos: BytePos) {
        // the new charpos must be > the last one (or it's the first one).
        let mut lines = self.lines.borrow_mut();
        let line_len = lines.len();
        assert!(line_len == 0 || ((*lines)[line_len - 1] < pos));
        lines.push(pos);
    }

    /// get a line from the list of pre-computed line-beginnings.
    /// line-number here is 0-based.
    pub fn get_line(&self, line_number: usize) -> Option<&str> {
        match self.src {
            Some(ref src) => {
                let lines = self.lines.borrow();
                lines.get(line_number).map(|&line| {
                    let begin: BytePos = line - self.start_pos;
                    let begin = begin.to_usize();
                    // We can't use `lines.get(line_number+1)` because we might
                    // be parsing when we call this function and thus the current
                    // line is the last one we have line info for.
                    let slice = &src[begin..];
                    match slice.find('\n') {
                        Some(e) => &slice[..e],
                        None => slice
                    }
                })
            }
            None => None
        }
    }

    pub fn record_multibyte_char(&self, pos: BytePos, bytes: usize) {
        assert!(bytes >=2 && bytes <= 4);
        let mbc = MultiByteChar {
            pos: pos,
            bytes: bytes,
        };
        self.multibyte_chars.borrow_mut().push(mbc);
    }

    pub fn is_real_file(&self) -> bool {
        !(self.name.starts_with("<") &&
          self.name.ends_with(">"))
    }

    pub fn is_imported(&self) -> bool {
        self.src.is_none()
    }

    pub fn byte_length(&self) -> u32 {
        self.end_pos.0 - self.start_pos.0
    }
    pub fn count_lines(&self) -> usize {
        self.lines.borrow().len()
    }

    /// Find the line containing the given position. The return value is the
    /// index into the `lines` array of this FileMap, not the 1-based line
    /// number. If the filemap is empty or the position is located before the
    /// first line, None is returned.
    pub fn lookup_line(&self, pos: BytePos) -> Option<usize> {
        let lines = self.lines.borrow();
        if lines.len() == 0 {
            return None;
        }

        let line_index = lookup_line(&lines[..], pos);
        assert!(line_index < lines.len() as isize);
        if line_index >= 0 {
            Some(line_index as usize)
        } else {
            None
        }
    }

    pub fn line_bounds(&self, line_index: usize) -> (BytePos, BytePos) {
        if self.start_pos == self.end_pos {
            return (self.start_pos, self.end_pos);
        }

        let lines = self.lines.borrow();
        assert!(line_index < lines.len());
        if line_index == (lines.len() - 1) {
            (lines[line_index], self.end_pos)
        } else {
            (lines[line_index], lines[line_index + 1])
        }
    }
}

// _____________________________________________________________________________
// Pos, BytePos, CharPos
//

pub trait Pos {
    fn from_usize(n: usize) -> Self;
    fn to_usize(&self) -> usize;
}

/// A byte offset. Keep this small (currently 32-bits), as AST contains
/// a lot of them.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct BytePos(pub u32);

/// A character offset. Because of multibyte utf8 characters, a byte offset
/// is not equivalent to a character offset. The CodeMap will convert BytePos
/// values to CharPos values as necessary.
#[derive(Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct CharPos(pub usize);

// FIXME: Lots of boilerplate in these impls, but so far my attempts to fix
// have been unsuccessful

impl Pos for BytePos {
    fn from_usize(n: usize) -> BytePos { BytePos(n as u32) }
    fn to_usize(&self) -> usize { let BytePos(n) = *self; n as usize }
}

impl Add for BytePos {
    type Output = BytePos;

    fn add(self, rhs: BytePos) -> BytePos {
        BytePos((self.to_usize() + rhs.to_usize()) as u32)
    }
}

impl Sub for BytePos {
    type Output = BytePos;

    fn sub(self, rhs: BytePos) -> BytePos {
        BytePos((self.to_usize() - rhs.to_usize()) as u32)
    }
}

impl Encodable for BytePos {
    fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        s.emit_u32(self.0)
    }
}

impl Decodable for BytePos {
    fn decode<D: Decoder>(d: &mut D) -> Result<BytePos, D::Error> {
        Ok(BytePos(d.read_u32()?))
    }
}

impl Pos for CharPos {
    fn from_usize(n: usize) -> CharPos { CharPos(n) }
    fn to_usize(&self) -> usize { let CharPos(n) = *self; n }
}

impl Add for CharPos {
    type Output = CharPos;

    fn add(self, rhs: CharPos) -> CharPos {
        CharPos(self.to_usize() + rhs.to_usize())
    }
}

impl Sub for CharPos {
    type Output = CharPos;

    fn sub(self, rhs: CharPos) -> CharPos {
        CharPos(self.to_usize() - rhs.to_usize())
    }
}

// _____________________________________________________________________________
// Loc, LocWithOpt, FileMapAndLine, FileMapAndBytePos
//

/// A source code location used for error reporting
#[derive(Debug, Clone)]
pub struct Loc {
    /// Information about the original source
    pub file: Rc<FileMap>,
    /// The (1-based) line number
    pub line: usize,
    /// The (0-based) column offset
    pub col: CharPos
}

/// A source code location used as the result of lookup_char_pos_adj
// Actually, *none* of the clients use the filename *or* file field;
// perhaps they should just be removed.
#[derive(Debug)]
pub struct LocWithOpt {
    pub filename: FileName,
    pub line: usize,
    pub col: CharPos,
    pub file: Option<Rc<FileMap>>,
}

// used to be structural records. Better names, anyone?
#[derive(Debug)]
pub struct FileMapAndLine { pub fm: Rc<FileMap>, pub line: usize }
#[derive(Debug)]
pub struct FileMapAndBytePos { pub fm: Rc<FileMap>, pub pos: BytePos }

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct LineInfo {
    /// Index of line, starting from 0.
    pub line_index: usize,

    /// Column in line where span begins, starting from 0.
    pub start_col: CharPos,

    /// Column in line where span ends, starting from 0, exclusive.
    pub end_col: CharPos,
}

pub struct FileLines {
    pub file: Rc<FileMap>,
    pub lines: Vec<LineInfo>
}

thread_local!(pub static SPAN_DEBUG: Cell<fn(Span, &mut fmt::Formatter) -> fmt::Result> =
                Cell::new(default_span_debug));

/* assuming that we're not in macro expansion */
pub fn mk_sp(lo: BytePos, hi: BytePos) -> Span {
    Span {lo: lo, hi: hi, expn_id: NO_EXPANSION}
}

pub struct MacroBacktrace {
    /// span where macro was applied to generate this code
    pub call_site: Span,

    /// name of macro that was applied (e.g., "foo!" or "#[derive(Eq)]")
    pub macro_decl_name: String,

    /// span where macro was defined (if known)
    pub def_site_span: Option<Span>,
}

// _____________________________________________________________________________
// SpanLinesError, SpanSnippetError, DistinctSources, MalformedCodemapPositions
//

pub type FileLinesResult = Result<FileLines, SpanLinesError>;

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum SpanLinesError {
    IllFormedSpan(Span),
    DistinctSources(DistinctSources),
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum SpanSnippetError {
    IllFormedSpan(Span),
    DistinctSources(DistinctSources),
    MalformedForCodemap(MalformedCodemapPositions),
    SourceNotAvailable { filename: String }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct DistinctSources {
    pub begin: (String, BytePos),
    pub end: (String, BytePos)
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct MalformedCodemapPositions {
    pub name: String,
    pub source_len: usize,
    pub begin_pos: BytePos,
    pub end_pos: BytePos
}

// Given a slice of line start positions and a position, returns the index of
// the line the position is on. Returns -1 if the position is located before
// the first line.
fn lookup_line(lines: &[BytePos], pos: BytePos) -> isize {
    match lines.binary_search(&pos) {
        Ok(line) => line as isize,
        Err(line) => line as isize - 1
    }
}

#[cfg(test)]
mod tests {
    use super::{lookup_line, BytePos};

    #[test]
    fn test_lookup_line() {

        let lines = &[BytePos(3), BytePos(17), BytePos(28)];

        assert_eq!(lookup_line(lines, BytePos(0)), -1);
        assert_eq!(lookup_line(lines, BytePos(3)),  0);
        assert_eq!(lookup_line(lines, BytePos(4)),  0);

        assert_eq!(lookup_line(lines, BytePos(16)), 0);
        assert_eq!(lookup_line(lines, BytePos(17)), 1);
        assert_eq!(lookup_line(lines, BytePos(18)), 1);

        assert_eq!(lookup_line(lines, BytePos(28)), 2);
        assert_eq!(lookup_line(lines, BytePos(29)), 2);
    }
}
