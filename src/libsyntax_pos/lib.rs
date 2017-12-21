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

#![doc(html_logo_url = "https://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
      html_favicon_url = "https://doc.rust-lang.org/favicon.ico",
      html_root_url = "https://doc.rust-lang.org/nightly/")]
#![deny(warnings)]

#![feature(const_fn)]
#![feature(custom_attribute)]
#![feature(i128_type)]
#![feature(optin_builtin_traits)]
#![allow(unused_attributes)]
#![feature(specialization)]

use std::borrow::Cow;
use std::cell::{Cell, RefCell};
use std::cmp::{self, Ordering};
use std::fmt;
use std::hash::{Hasher, Hash};
use std::ops::{Add, Sub};
use std::path::PathBuf;
use std::rc::Rc;

use rustc_data_structures::stable_hasher::StableHasher;

extern crate rustc_data_structures;

use serialize::{Encodable, Decodable, Encoder, Decoder};

extern crate serialize;
extern crate serialize as rustc_serialize; // used by deriving

extern crate unicode_width;

pub mod hygiene;
pub use hygiene::{SyntaxContext, ExpnInfo, ExpnFormat, NameAndSpan, CompilerDesugaringKind};

mod span_encoding;
pub use span_encoding::{Span, DUMMY_SP};

pub mod symbol;

/// Differentiates between real files and common virtual files
#[derive(Debug, Eq, PartialEq, Clone, Ord, PartialOrd, Hash, RustcDecodable, RustcEncodable)]
pub enum FileName {
    Real(PathBuf),
    /// e.g. "std" macros
    Macros(String),
    /// call to `quote!`
    QuoteExpansion,
    /// Command line
    Anon,
    /// Hack in src/libsyntax/parse.rs
    /// FIXME(jseyfried)
    MacroExpansion,
    ProcMacroSourceCode,
    /// Strings provided as --cfg [cfgspec] stored in a crate_cfg
    CfgSpec,
    /// Custom sources for explicit parser calls from plugins and drivers
    Custom(String),
}

impl std::fmt::Display for FileName {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        use self::FileName::*;
        match *self {
            Real(ref path) => write!(fmt, "{}", path.display()),
            Macros(ref name) => write!(fmt, "<{} macros>", name),
            QuoteExpansion => write!(fmt, "<quote expansion>"),
            MacroExpansion => write!(fmt, "<macro expansion>"),
            Anon => write!(fmt, "<anon>"),
            ProcMacroSourceCode => write!(fmt, "<proc-macro source code>"),
            CfgSpec => write!(fmt, "cfgspec"),
            Custom(ref s) => write!(fmt, "<{}>", s),
        }
    }
}

impl From<PathBuf> for FileName {
    fn from(p: PathBuf) -> Self {
        assert!(!p.to_string_lossy().ends_with('>'));
        FileName::Real(p)
    }
}

impl FileName {
    pub fn is_real(&self) -> bool {
        use self::FileName::*;
        match *self {
            Real(_) => true,
            Macros(_) |
            Anon |
            MacroExpansion |
            ProcMacroSourceCode |
            CfgSpec |
            Custom(_) |
            QuoteExpansion => false,
        }
    }

    pub fn is_macros(&self) -> bool {
        use self::FileName::*;
        match *self {
            Real(_) |
            Anon |
            MacroExpansion |
            ProcMacroSourceCode |
            CfgSpec |
            Custom(_) |
            QuoteExpansion => false,
            Macros(_) => true,
        }
    }
}

/// Spans represent a region of code, used for error reporting. Positions in spans
/// are *absolute* positions from the beginning of the codemap, not positions
/// relative to FileMaps. Methods on the CodeMap can be used to relate spans back
/// to the original source.
/// You must be careful if the span crosses more than one file - you will not be
/// able to use many of the functions on spans in codemap and you cannot assume
/// that the length of the span = hi - lo; there may be space in the BytePos
/// range between files.
///
/// `SpanData` is public because `Span` uses a thread-local interner and can't be
/// sent to other threads, but some pieces of performance infra run in a separate thread.
/// Using `Span` is generally preferred.
#[derive(Clone, Copy, Hash, PartialEq, Eq, Ord, PartialOrd)]
pub struct SpanData {
    pub lo: BytePos,
    pub hi: BytePos,
    /// Information about where the macro came from, if this piece of
    /// code was created by a macro expansion.
    pub ctxt: SyntaxContext,
}

impl SpanData {
    #[inline]
    pub fn with_lo(&self, lo: BytePos) -> Span {
        Span::new(lo, self.hi, self.ctxt)
    }
    #[inline]
    pub fn with_hi(&self, hi: BytePos) -> Span {
        Span::new(self.lo, hi, self.ctxt)
    }
    #[inline]
    pub fn with_ctxt(&self, ctxt: SyntaxContext) -> Span {
        Span::new(self.lo, self.hi, ctxt)
    }
}

// The interner in thread-local, so `Span` shouldn't move between threads.
impl !Send for Span {}
impl !Sync for Span {}

impl PartialOrd for Span {
    fn partial_cmp(&self, rhs: &Self) -> Option<Ordering> {
        PartialOrd::partial_cmp(&self.data(), &rhs.data())
    }
}
impl Ord for Span {
    fn cmp(&self, rhs: &Self) -> Ordering {
        Ord::cmp(&self.data(), &rhs.data())
    }
}

/// A collection of spans. Spans have two orthogonal attributes:
///
/// - they can be *primary spans*. In this case they are the locus of
///   the error, and would be rendered with `^^^`.
/// - they can have a *label*. In this case, the label is written next
///   to the mark in the snippet when we render.
#[derive(Clone, Debug, Hash, PartialEq, Eq, RustcEncodable, RustcDecodable)]
pub struct MultiSpan {
    primary_spans: Vec<Span>,
    span_labels: Vec<(Span, String)>,
}

impl Span {
    #[inline]
    pub fn lo(self) -> BytePos {
        self.data().lo
    }
    #[inline]
    pub fn with_lo(self, lo: BytePos) -> Span {
        self.data().with_lo(lo)
    }
    #[inline]
    pub fn hi(self) -> BytePos {
        self.data().hi
    }
    #[inline]
    pub fn with_hi(self, hi: BytePos) -> Span {
        self.data().with_hi(hi)
    }
    #[inline]
    pub fn ctxt(self) -> SyntaxContext {
        self.data().ctxt
    }
    #[inline]
    pub fn with_ctxt(self, ctxt: SyntaxContext) -> Span {
        self.data().with_ctxt(ctxt)
    }

    /// Returns a new span representing just the end-point of this span
    pub fn end_point(self) -> Span {
        let span = self.data();
        let lo = cmp::max(span.hi.0 - 1, span.lo.0);
        span.with_lo(BytePos(lo))
    }

    /// Returns a new span representing the next character after the end-point of this span
    pub fn next_point(self) -> Span {
        let span = self.data();
        let lo = cmp::max(span.hi.0, span.lo.0 + 1);
        Span::new(BytePos(lo), BytePos(lo), span.ctxt)
    }

    /// Returns `self` if `self` is not the dummy span, and `other` otherwise.
    pub fn substitute_dummy(self, other: Span) -> Span {
        if self.source_equal(&DUMMY_SP) { other } else { self }
    }

    /// Return true if `self` fully encloses `other`.
    pub fn contains(self, other: Span) -> bool {
        let span = self.data();
        let other = other.data();
        span.lo <= other.lo && other.hi <= span.hi
    }

    /// Return true if the spans are equal with regards to the source text.
    ///
    /// Use this instead of `==` when either span could be generated code,
    /// and you only care that they point to the same bytes of source text.
    pub fn source_equal(&self, other: &Span) -> bool {
        let span = self.data();
        let other = other.data();
        span.lo == other.lo && span.hi == other.hi
    }

    /// Returns `Some(span)`, where the start is trimmed by the end of `other`
    pub fn trim_start(self, other: Span) -> Option<Span> {
        let span = self.data();
        let other = other.data();
        if span.hi > other.hi {
            Some(span.with_lo(cmp::max(span.lo, other.hi)))
        } else {
            None
        }
    }

    /// Return the source span - this is either the supplied span, or the span for
    /// the macro callsite that expanded to it.
    pub fn source_callsite(self) -> Span {
        self.ctxt().outer().expn_info().map(|info| info.call_site.source_callsite()).unwrap_or(self)
    }

    /// Return the source callee.
    ///
    /// Returns None if the supplied span has no expansion trace,
    /// else returns the NameAndSpan for the macro definition
    /// corresponding to the source callsite.
    pub fn source_callee(self) -> Option<NameAndSpan> {
        fn source_callee(info: ExpnInfo) -> NameAndSpan {
            match info.call_site.ctxt().outer().expn_info() {
                Some(info) => source_callee(info),
                None => info.callee,
            }
        }
        self.ctxt().outer().expn_info().map(source_callee)
    }

    /// Check if a span is "internal" to a macro in which #[unstable]
    /// items can be used (that is, a macro marked with
    /// `#[allow_internal_unstable]`).
    pub fn allows_unstable(&self) -> bool {
        match self.ctxt().outer().expn_info() {
            Some(info) => info.callee.allow_internal_unstable,
            None => false,
        }
    }

    /// Check if this span arises from a compiler desugaring of kind `kind`.
    pub fn is_compiler_desugaring(&self, kind: CompilerDesugaringKind) -> bool {
        match self.ctxt().outer().expn_info() {
            Some(info) => match info.callee.format {
                ExpnFormat::CompilerDesugaring(k) => k == kind,
                _ => false,
            },
            None => false,
        }
    }

    /// Return the compiler desugaring that created this span, or None
    /// if this span is not from a desugaring.
    pub fn compiler_desugaring_kind(&self) -> Option<CompilerDesugaringKind> {
        match self.ctxt().outer().expn_info() {
            Some(info) => match info.callee.format {
                ExpnFormat::CompilerDesugaring(k) => Some(k),
                _ => None
            },
            None => None
        }
    }

    /// Check if a span is "internal" to a macro in which `unsafe`
    /// can be used without triggering the `unsafe_code` lint
    //  (that is, a macro marked with `#[allow_internal_unsafe]`).
    pub fn allows_unsafe(&self) -> bool {
        match self.ctxt().outer().expn_info() {
            Some(info) => info.callee.allow_internal_unsafe,
            None => false,
        }
    }

    pub fn macro_backtrace(mut self) -> Vec<MacroBacktrace> {
        let mut prev_span = DUMMY_SP;
        let mut result = vec![];
        loop {
            let info = match self.ctxt().outer().expn_info() {
                Some(info) => info,
                None => break,
            };

            let (pre, post) = match info.callee.format {
                ExpnFormat::MacroAttribute(..) => ("#[", "]"),
                ExpnFormat::MacroBang(..) => ("", "!"),
                ExpnFormat::CompilerDesugaring(..) => ("desugaring of `", "`"),
            };
            let macro_decl_name = format!("{}{}{}", pre, info.callee.name(), post);
            let def_site_span = info.callee.span;

            // Don't print recursive invocations
            if !info.call_site.source_equal(&prev_span) {
                result.push(MacroBacktrace {
                    call_site: info.call_site,
                    macro_decl_name,
                    def_site_span,
                });
            }

            prev_span = self;
            self = info.call_site;
        }
        result
    }

    /// Return a `Span` that would enclose both `self` and `end`.
    pub fn to(self, end: Span) -> Span {
        let span = self.data();
        let end = end.data();
        Span::new(
            cmp::min(span.lo, end.lo),
            cmp::max(span.hi, end.hi),
            // FIXME(jseyfried): self.ctxt should always equal end.ctxt here (c.f. issue #23480)
            if span.ctxt == SyntaxContext::empty() { end.ctxt } else { span.ctxt },
        )
    }

    /// Return a `Span` between the end of `self` to the beginning of `end`.
    pub fn between(self, end: Span) -> Span {
        let span = self.data();
        let end = end.data();
        Span::new(
            span.hi,
            end.lo,
            if end.ctxt == SyntaxContext::empty() { end.ctxt } else { span.ctxt },
        )
    }

    /// Return a `Span` between the beginning of `self` to the beginning of `end`.
    pub fn until(self, end: Span) -> Span {
        let span = self.data();
        let end = end.data();
        Span::new(
            span.lo,
            end.lo,
            if end.ctxt == SyntaxContext::empty() { end.ctxt } else { span.ctxt },
        )
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

impl Default for Span {
    fn default() -> Self {
        DUMMY_SP
    }
}

impl serialize::UseSpecializedEncodable for Span {
    fn default_encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        let span = self.data();
        s.emit_struct("Span", 2, |s| {
            s.emit_struct_field("lo", 0, |s| {
                span.lo.encode(s)
            })?;

            s.emit_struct_field("hi", 1, |s| {
                span.hi.encode(s)
            })
        })
    }
}

impl serialize::UseSpecializedDecodable for Span {
    fn default_decode<D: Decoder>(d: &mut D) -> Result<Span, D::Error> {
        d.read_struct("Span", 2, |d| {
            let lo = d.read_struct_field("lo", 0, Decodable::decode)?;
            let hi = d.read_struct_field("hi", 1, Decodable::decode)?;
            Ok(Span::new(lo, hi, NO_EXPANSION))
        })
    }
}

fn default_span_debug(span: Span, f: &mut fmt::Formatter) -> fmt::Result {
    f.debug_struct("Span")
        .field("lo", &span.lo())
        .field("hi", &span.hi())
        .field("ctxt", &span.ctxt())
        .finish()
}

impl fmt::Debug for Span {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        SPAN_DEBUG.with(|span_debug| span_debug.get()(*self, f))
    }
}

impl fmt::Debug for SpanData {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        SPAN_DEBUG.with(|span_debug| span_debug.get()(Span::new(self.lo, self.hi, self.ctxt), f))
    }
}

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

    /// Replaces all occurrences of one Span with another. Used to move Spans in areas that don't
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
                span,
                is_primary: is_primary(span),
                label: Some(label.clone())
            });
        }

        for &span in &self.primary_spans {
            if !span_labels.iter().any(|sl| sl.span == span) {
                span_labels.push(SpanLabel {
                    span,
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

impl From<Vec<Span>> for MultiSpan {
    fn from(spans: Vec<Span>) -> MultiSpan {
        MultiSpan::from_spans(spans)
    }
}

pub const NO_EXPANSION: SyntaxContext = SyntaxContext::empty();

/// Identifies an offset of a multi-byte character in a FileMap
#[derive(Copy, Clone, RustcEncodable, RustcDecodable, Eq, PartialEq)]
pub struct MultiByteChar {
    /// The absolute offset of the character in the CodeMap
    pub pos: BytePos,
    /// The number of bytes, >=2
    pub bytes: usize,
}

/// Identifies an offset of a non-narrow character in a FileMap
#[derive(Copy, Clone, RustcEncodable, RustcDecodable, Eq, PartialEq)]
pub enum NonNarrowChar {
    /// Represents a zero-width character
    ZeroWidth(BytePos),
    /// Represents a wide (fullwidth) character
    Wide(BytePos),
    /// Represents a tab character, represented visually with a width of 4 characters
    Tab(BytePos),
}

impl NonNarrowChar {
    fn new(pos: BytePos, width: usize) -> Self {
        match width {
            0 => NonNarrowChar::ZeroWidth(pos),
            2 => NonNarrowChar::Wide(pos),
            4 => NonNarrowChar::Tab(pos),
            _ => panic!("width {} given for non-narrow character", width),
        }
    }

    /// Returns the absolute offset of the character in the CodeMap
    pub fn pos(&self) -> BytePos {
        match *self {
            NonNarrowChar::ZeroWidth(p) |
            NonNarrowChar::Wide(p) |
            NonNarrowChar::Tab(p) => p,
        }
    }

    /// Returns the width of the character, 0 (zero-width) or 2 (wide)
    pub fn width(&self) -> usize {
        match *self {
            NonNarrowChar::ZeroWidth(_) => 0,
            NonNarrowChar::Wide(_) => 2,
            NonNarrowChar::Tab(_) => 4,
        }
    }
}

impl Add<BytePos> for NonNarrowChar {
    type Output = Self;

    fn add(self, rhs: BytePos) -> Self {
        match self {
            NonNarrowChar::ZeroWidth(pos) => NonNarrowChar::ZeroWidth(pos + rhs),
            NonNarrowChar::Wide(pos) => NonNarrowChar::Wide(pos + rhs),
            NonNarrowChar::Tab(pos) => NonNarrowChar::Tab(pos + rhs),
        }
    }
}

impl Sub<BytePos> for NonNarrowChar {
    type Output = Self;

    fn sub(self, rhs: BytePos) -> Self {
        match self {
            NonNarrowChar::ZeroWidth(pos) => NonNarrowChar::ZeroWidth(pos - rhs),
            NonNarrowChar::Wide(pos) => NonNarrowChar::Wide(pos - rhs),
            NonNarrowChar::Tab(pos) => NonNarrowChar::Tab(pos - rhs),
        }
    }
}

/// The state of the lazy external source loading mechanism of a FileMap.
#[derive(PartialEq, Eq, Clone)]
pub enum ExternalSource {
    /// The external source has been loaded already.
    Present(String),
    /// No attempt has been made to load the external source.
    AbsentOk,
    /// A failed attempt has been made to load the external source.
    AbsentErr,
    /// No external source has to be loaded, since the FileMap represents a local crate.
    Unneeded,
}

impl ExternalSource {
    pub fn is_absent(&self) -> bool {
        match *self {
            ExternalSource::Present(_) => false,
            _ => true,
        }
    }

    pub fn get_source(&self) -> Option<&str> {
        match *self {
            ExternalSource::Present(ref src) => Some(src),
            _ => None,
        }
    }
}

/// A single source in the CodeMap.
#[derive(Clone)]
pub struct FileMap {
    /// The name of the file that the source came from, source that doesn't
    /// originate from files has names between angle brackets by convention,
    /// e.g. `<anon>`
    pub name: FileName,
    /// True if the `name` field above has been modified by -Zremap-path-prefix
    pub name_was_remapped: bool,
    /// The unmapped path of the file that the source came from.
    /// Set to `None` if the FileMap was imported from an external crate.
    pub unmapped_path: Option<FileName>,
    /// Indicates which crate this FileMap was imported from.
    pub crate_of_origin: u32,
    /// The complete source code
    pub src: Option<Rc<String>>,
    /// The source code's hash
    pub src_hash: u128,
    /// The external source code (used for external crates, which will have a `None`
    /// value as `self.src`.
    pub external_src: RefCell<ExternalSource>,
    /// The start position of this source in the CodeMap
    pub start_pos: BytePos,
    /// The end position of this source in the CodeMap
    pub end_pos: BytePos,
    /// Locations of lines beginnings in the source code
    pub lines: RefCell<Vec<BytePos>>,
    /// Locations of multi-byte characters in the source code
    pub multibyte_chars: RefCell<Vec<MultiByteChar>>,
    /// Width of characters that are not narrow in the source code
    pub non_narrow_chars: RefCell<Vec<NonNarrowChar>>,
    /// A hash of the filename, used for speeding up the incr. comp. hashing.
    pub name_hash: u128,
}

impl Encodable for FileMap {
    fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        s.emit_struct("FileMap", 8, |s| {
            s.emit_struct_field("name", 0, |s| self.name.encode(s))?;
            s.emit_struct_field("name_was_remapped", 1, |s| self.name_was_remapped.encode(s))?;
            s.emit_struct_field("src_hash", 2, |s| self.src_hash.encode(s))?;
            s.emit_struct_field("start_pos", 4, |s| self.start_pos.encode(s))?;
            s.emit_struct_field("end_pos", 5, |s| self.end_pos.encode(s))?;
            s.emit_struct_field("lines", 6, |s| {
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
            s.emit_struct_field("multibyte_chars", 7, |s| {
                (*self.multibyte_chars.borrow()).encode(s)
            })?;
            s.emit_struct_field("non_narrow_chars", 8, |s| {
                (*self.non_narrow_chars.borrow()).encode(s)
            })?;
            s.emit_struct_field("name_hash", 9, |s| {
                self.name_hash.encode(s)
            })
        })
    }
}

impl Decodable for FileMap {
    fn decode<D: Decoder>(d: &mut D) -> Result<FileMap, D::Error> {

        d.read_struct("FileMap", 8, |d| {
            let name: FileName = d.read_struct_field("name", 0, |d| Decodable::decode(d))?;
            let name_was_remapped: bool =
                d.read_struct_field("name_was_remapped", 1, |d| Decodable::decode(d))?;
            let src_hash: u128 =
                d.read_struct_field("src_hash", 2, |d| Decodable::decode(d))?;
            let start_pos: BytePos =
                d.read_struct_field("start_pos", 4, |d| Decodable::decode(d))?;
            let end_pos: BytePos = d.read_struct_field("end_pos", 5, |d| Decodable::decode(d))?;
            let lines: Vec<BytePos> = d.read_struct_field("lines", 6, |d| {
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
                d.read_struct_field("multibyte_chars", 7, |d| Decodable::decode(d))?;
            let non_narrow_chars: Vec<NonNarrowChar> =
                d.read_struct_field("non_narrow_chars", 8, |d| Decodable::decode(d))?;
            let name_hash: u128 =
                d.read_struct_field("name_hash", 9, |d| Decodable::decode(d))?;
            Ok(FileMap {
                name,
                name_was_remapped,
                unmapped_path: None,
                // `crate_of_origin` has to be set by the importer.
                // This value matches up with rustc::hir::def_id::INVALID_CRATE.
                // That constant is not available here unfortunately :(
                crate_of_origin: ::std::u32::MAX - 1,
                start_pos,
                end_pos,
                src: None,
                src_hash,
                external_src: RefCell::new(ExternalSource::AbsentOk),
                lines: RefCell::new(lines),
                multibyte_chars: RefCell::new(multibyte_chars),
                non_narrow_chars: RefCell::new(non_narrow_chars),
                name_hash,
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
    pub fn new(name: FileName,
               name_was_remapped: bool,
               unmapped_path: FileName,
               mut src: String,
               start_pos: BytePos) -> FileMap {
        remove_bom(&mut src);

        let src_hash = {
            let mut hasher: StableHasher<u128> = StableHasher::new();
            hasher.write(src.as_bytes());
            hasher.finish()
        };
        let name_hash = {
            let mut hasher: StableHasher<u128> = StableHasher::new();
            name.hash(&mut hasher);
            hasher.finish()
        };
        let end_pos = start_pos.to_usize() + src.len();

        FileMap {
            name,
            name_was_remapped,
            unmapped_path: Some(unmapped_path),
            crate_of_origin: 0,
            src: Some(Rc::new(src)),
            src_hash,
            external_src: RefCell::new(ExternalSource::Unneeded),
            start_pos,
            end_pos: Pos::from_usize(end_pos),
            lines: RefCell::new(Vec::new()),
            multibyte_chars: RefCell::new(Vec::new()),
            non_narrow_chars: RefCell::new(Vec::new()),
            name_hash,
        }
    }

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

    /// Add externally loaded source.
    /// If the hash of the input doesn't match or no input is supplied via None,
    /// it is interpreted as an error and the corresponding enum variant is set.
    /// The return value signifies whether some kind of source is present.
    pub fn add_external_src<F>(&self, get_src: F) -> bool
        where F: FnOnce() -> Option<String>
    {
        if *self.external_src.borrow() == ExternalSource::AbsentOk {
            let src = get_src();
            let mut external_src = self.external_src.borrow_mut();
            if let Some(src) = src {
                let mut hasher: StableHasher<u128> = StableHasher::new();
                hasher.write(src.as_bytes());

                if hasher.finish() == self.src_hash {
                    *external_src = ExternalSource::Present(src);
                    return true;
                }
            } else {
                *external_src = ExternalSource::AbsentErr;
            }

            false
        } else {
            self.src.is_some() || self.external_src.borrow().get_source().is_some()
        }
    }

    /// Get a line from the list of pre-computed line-beginnings.
    /// The line number here is 0-based.
    pub fn get_line(&self, line_number: usize) -> Option<Cow<str>> {
        fn get_until_newline(src: &str, begin: usize) -> &str {
            // We can't use `lines.get(line_number+1)` because we might
            // be parsing when we call this function and thus the current
            // line is the last one we have line info for.
            let slice = &src[begin..];
            match slice.find('\n') {
                Some(e) => &slice[..e],
                None => slice
            }
        }

        let lines = self.lines.borrow();
        let line = if let Some(line) = lines.get(line_number) {
            line
        } else {
            return None;
        };
        let begin: BytePos = *line - self.start_pos;
        let begin = begin.to_usize();

        if let Some(ref src) = self.src {
            Some(Cow::from(get_until_newline(src, begin)))
        } else if let Some(src) = self.external_src.borrow().get_source() {
            Some(Cow::Owned(String::from(get_until_newline(src, begin))))
        } else {
            None
        }
    }

    pub fn record_multibyte_char(&self, pos: BytePos, bytes: usize) {
        assert!(bytes >=2 && bytes <= 4);
        let mbc = MultiByteChar {
            pos,
            bytes,
        };
        self.multibyte_chars.borrow_mut().push(mbc);
    }

    pub fn record_width(&self, pos: BytePos, ch: char) {
        let width = match ch {
            '\t' =>
                // Tabs will consume 4 columns.
                4,
            '\n' =>
                // Make newlines take one column so that displayed spans can point them.
                1,
            ch =>
                // Assume control characters are zero width.
                // FIXME: How can we decide between `width` and `width_cjk`?
                unicode_width::UnicodeWidthChar::width(ch).unwrap_or(0),
        };
        // Only record non-narrow characters.
        if width != 1 {
            self.non_narrow_chars.borrow_mut().push(NonNarrowChar::new(pos, width));
        }
    }

    pub fn is_real_file(&self) -> bool {
        self.name.is_real()
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

    #[inline]
    pub fn contains(&self, byte_pos: BytePos) -> bool {
        byte_pos >= self.start_pos && byte_pos <= self.end_pos
    }
}

/// Remove utf-8 BOM if any.
fn remove_bom(src: &mut String) {
    if src.starts_with("\u{feff}") {
        src.drain(..3);
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
    pub col: CharPos,
    /// The (0-based) column offset when displayed
    pub col_display: usize,
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

#[derive(Debug)]
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
    SourceNotAvailable { filename: FileName }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct DistinctSources {
    pub begin: (FileName, BytePos),
    pub end: (FileName, BytePos)
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct MalformedCodemapPositions {
    pub name: FileName,
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
