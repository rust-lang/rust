// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! The CodeMap tracks all the source code used within a single crate, mapping
//! from integer byte positions to the original source code location. Each bit
//! of source parsed during crate parsing (typically files, in-memory strings,
//! or various bits of macro expansion) cover a continuous range of bytes in the
//! CodeMap and are represented by FileMaps. Byte positions are stored in
//! `spans` and used pervasively in the compiler. They are absolute positions
//! within the CodeMap, which upon request can be converted to line and column
//! information, source code snippets, etc.

pub use self::ExpnFormat::*;

use std::cell::{Cell, RefCell};
use std::ops::{Add, Sub};
use std::path::Path;
use std::rc::Rc;

use std::{fmt, fs};
use std::io::{self, Read};

use serialize::{Encodable, Decodable, Encoder, Decoder};

use ast::Name;

// _____________________________________________________________________________
// Pos, BytePos, CharPos
//

pub trait Pos {
    fn from_usize(n: usize) -> Self;
    fn to_usize(&self) -> usize;
}

/// A byte offset. Keep this small (currently 32-bits), as AST contains
/// a lot of them.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Debug)]
pub struct BytePos(pub u32);

/// A character offset. Because of multibyte utf8 characters, a byte offset
/// is not equivalent to a character offset. The CodeMap will convert BytePos
/// values to CharPos values as necessary.
#[derive(Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Debug)]
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
        Ok(BytePos(try!{ d.read_u32() }))
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
// Span, Spanned
//

/// Spans represent a region of code, used for error reporting. Positions in spans
/// are *absolute* positions from the beginning of the codemap, not positions
/// relative to FileMaps. Methods on the CodeMap can be used to relate spans back
/// to the original source.
/// You must be careful if the span crosses more than one file - you will not be
/// able to use many of the functions on spans in codemap and you cannot assume
/// that the length of the span = hi - lo; there may be space in the BytePos
/// range between files.
#[derive(Clone, Copy, Hash)]
pub struct Span {
    pub lo: BytePos,
    pub hi: BytePos,
    /// Information about where the macro came from, if this piece of
    /// code was created by a macro expansion.
    pub expn_id: ExpnId
}

pub const DUMMY_SP: Span = Span { lo: BytePos(0), hi: BytePos(0), expn_id: NO_EXPANSION };

// Generic span to be used for code originating from the command line
pub const COMMAND_LINE_SP: Span = Span { lo: BytePos(0),
                                         hi: BytePos(0),
                                         expn_id: COMMAND_LINE_EXPN };

impl Span {
    /// Returns `self` if `self` is not the dummy span, and `other` otherwise.
    pub fn substitute_dummy(self, other: Span) -> Span {
        if self == DUMMY_SP { other } else { self }
    }

    pub fn contains(self, other: Span) -> bool {
        self.lo <= other.lo && other.hi <= self.hi
    }
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug, Copy)]
pub struct Spanned<T> {
    pub node: T,
    pub span: Span,
}

impl PartialEq for Span {
    fn eq(&self, other: &Span) -> bool {
        return (*self).lo == (*other).lo && (*self).hi == (*other).hi;
    }
    fn ne(&self, other: &Span) -> bool { !(*self).eq(other) }
}

impl Eq for Span {}

impl Encodable for Span {
    fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        try!(s.emit_u32(self.lo.0));
        s.emit_u32(self.hi.0)
    }
}

impl Decodable for Span {
    fn decode<D: Decoder>(d: &mut D) -> Result<Span, D::Error> {
        let lo = BytePos(try! { d.read_u32() });
        let hi = BytePos(try! { d.read_u32() });
        Ok(mk_sp(lo, hi))
    }
}

fn default_span_debug(span: Span, f: &mut fmt::Formatter) -> fmt::Result {
    write!(f, "Span {{ lo: {:?}, hi: {:?}, expn_id: {:?} }}",
           span.lo, span.hi, span.expn_id)
}

thread_local!(pub static SPAN_DEBUG: Cell<fn(Span, &mut fmt::Formatter) -> fmt::Result> =
                Cell::new(default_span_debug));

impl fmt::Debug for Span {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        SPAN_DEBUG.with(|span_debug| span_debug.get()(*self, f))
    }
}

pub fn spanned<T>(lo: BytePos, hi: BytePos, t: T) -> Spanned<T> {
    respan(mk_sp(lo, hi), t)
}

pub fn respan<T>(sp: Span, t: T) -> Spanned<T> {
    Spanned {node: t, span: sp}
}

pub fn dummy_spanned<T>(t: T) -> Spanned<T> {
    respan(DUMMY_SP, t)
}

/* assuming that we're not in macro expansion */
pub fn mk_sp(lo: BytePos, hi: BytePos) -> Span {
    Span {lo: lo, hi: hi, expn_id: NO_EXPANSION}
}

/// Return the span itself if it doesn't come from a macro expansion,
/// otherwise return the call site span up to the `enclosing_sp` by
/// following the `expn_info` chain.
pub fn original_sp(cm: &CodeMap, sp: Span, enclosing_sp: Span) -> Span {
    let call_site1 = cm.with_expn_info(sp.expn_id, |ei| ei.map(|ei| ei.call_site));
    let call_site2 = cm.with_expn_info(enclosing_sp.expn_id, |ei| ei.map(|ei| ei.call_site));
    match (call_site1, call_site2) {
        (None, _) => sp,
        (Some(call_site1), Some(call_site2)) if call_site1 == call_site2 => sp,
        (Some(call_site1), _) => original_sp(cm, call_site1, enclosing_sp),
    }
}

// _____________________________________________________________________________
// Loc, LocWithOpt, FileMapAndLine, FileMapAndBytePos
//

/// A source code location used for error reporting
#[derive(Debug)]
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


// _____________________________________________________________________________
// ExpnFormat, NameAndSpan, ExpnInfo, ExpnId
//

/// The source of expansion.
#[derive(Clone, Hash, Debug, PartialEq, Eq)]
pub enum ExpnFormat {
    /// e.g. #[derive(...)] <item>
    MacroAttribute(Name),
    /// e.g. `format!()`
    MacroBang(Name),
}

#[derive(Clone, Hash, Debug)]
pub struct NameAndSpan {
    /// The format with which the macro was invoked.
    pub format: ExpnFormat,
    /// Whether the macro is allowed to use #[unstable]/feature-gated
    /// features internally without forcing the whole crate to opt-in
    /// to them.
    pub allow_internal_unstable: bool,
    /// The span of the macro definition itself. The macro may not
    /// have a sensible definition span (e.g. something defined
    /// completely inside libsyntax) in which case this is None.
    pub span: Option<Span>
}

impl NameAndSpan {
    pub fn name(&self) -> Name {
        match self.format {
            ExpnFormat::MacroAttribute(s) => s,
            ExpnFormat::MacroBang(s) => s,
        }
    }
}

/// Extra information for tracking spans of macro and syntax sugar expansion
#[derive(Hash, Debug)]
pub struct ExpnInfo {
    /// The location of the actual macro invocation or syntax sugar , e.g.
    /// `let x = foo!();` or `if let Some(y) = x {}`
    ///
    /// This may recursively refer to other macro invocations, e.g. if
    /// `foo!()` invoked `bar!()` internally, and there was an
    /// expression inside `bar!`; the call_site of the expression in
    /// the expansion would point to the `bar!` invocation; that
    /// call_site span would have its own ExpnInfo, with the call_site
    /// pointing to the `foo!` invocation.
    pub call_site: Span,
    /// Information about the expansion.
    pub callee: NameAndSpan
}

#[derive(PartialEq, Eq, Clone, Debug, Hash, RustcEncodable, RustcDecodable, Copy)]
pub struct ExpnId(u32);

pub const NO_EXPANSION: ExpnId = ExpnId(!0);
// For code appearing from the command line
pub const COMMAND_LINE_EXPN: ExpnId = ExpnId(!1);

impl ExpnId {
    pub fn from_u32(id: u32) -> ExpnId {
        ExpnId(id)
    }

    pub fn into_u32(self) -> u32 {
        self.0
    }
}

// _____________________________________________________________________________
// FileMap, MultiByteChar, FileName, FileLines
//

pub type FileName = String;

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
        s.emit_struct("FileMap", 5, |s| {
            try! { s.emit_struct_field("name", 0, |s| self.name.encode(s)) };
            try! { s.emit_struct_field("start_pos", 1, |s| self.start_pos.encode(s)) };
            try! { s.emit_struct_field("end_pos", 2, |s| self.end_pos.encode(s)) };
            try! { s.emit_struct_field("lines", 3, |s| {
                    let lines = self.lines.borrow();
                    // store the length
                    try! { s.emit_u32(lines.len() as u32) };

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
                        try! { bytes_per_diff.encode(s) };

                        // Encode the first element.
                        try! { lines[0].encode(s) };

                        let diff_iter = (&lines[..]).windows(2)
                                                    .map(|w| (w[1] - w[0]));

                        match bytes_per_diff {
                            1 => for diff in diff_iter { try! { (diff.0 as u8).encode(s) } },
                            2 => for diff in diff_iter { try! { (diff.0 as u16).encode(s) } },
                            4 => for diff in diff_iter { try! { diff.0.encode(s) } },
                            _ => unreachable!()
                        }
                    }

                    Ok(())
                })
            };
            s.emit_struct_field("multibyte_chars", 4, |s| {
                (*self.multibyte_chars.borrow()).encode(s)
            })
        })
    }
}

impl Decodable for FileMap {
    fn decode<D: Decoder>(d: &mut D) -> Result<FileMap, D::Error> {

        d.read_struct("FileMap", 5, |d| {
            let name: String = try! {
                d.read_struct_field("name", 0, |d| Decodable::decode(d))
            };
            let start_pos: BytePos = try! {
                d.read_struct_field("start_pos", 1, |d| Decodable::decode(d))
            };
            let end_pos: BytePos = try! {
                d.read_struct_field("end_pos", 2, |d| Decodable::decode(d))
            };
            let lines: Vec<BytePos> = try! {
                d.read_struct_field("lines", 3, |d| {
                    let num_lines: u32 = try! { Decodable::decode(d) };
                    let mut lines = Vec::with_capacity(num_lines as usize);

                    if num_lines > 0 {
                        // Read the number of bytes used per diff.
                        let bytes_per_diff: u8 = try! { Decodable::decode(d) };

                        // Read the first element.
                        let mut line_start: BytePos = try! { Decodable::decode(d) };
                        lines.push(line_start);

                        for _ in 1..num_lines {
                            let diff = match bytes_per_diff {
                                1 => try! { d.read_u8() } as u32,
                                2 => try! { d.read_u16() } as u32,
                                4 => try! { d.read_u32() },
                                _ => unreachable!()
                            };

                            line_start = line_start + BytePos(diff);

                            lines.push(line_start);
                        }
                    }

                    Ok(lines)
                })
            };
            let multibyte_chars: Vec<MultiByteChar> = try! {
                d.read_struct_field("multibyte_chars", 4, |d| Decodable::decode(d))
            };
            Ok(FileMap {
                name: name,
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

    fn count_lines(&self) -> usize {
        self.lines.borrow().len()
    }
}

/// An abstraction over the fs operations used by the Parser.
pub trait FileLoader {
    /// Query the existence of a file.
    fn file_exists(&self, path: &Path) -> bool;

    /// Read the contents of an UTF-8 file into memory.
    fn read_file(&self, path: &Path) -> io::Result<String>;
}

/// A FileLoader that uses std::fs to load real files.
pub struct RealFileLoader;

impl FileLoader for RealFileLoader {
    fn file_exists(&self, path: &Path) -> bool {
        fs::metadata(path).is_ok()
    }

    fn read_file(&self, path: &Path) -> io::Result<String> {
        let mut src = String::new();
        try!(try!(fs::File::open(path)).read_to_string(&mut src));
        Ok(src)
    }
}

// _____________________________________________________________________________
// CodeMap
//

pub struct CodeMap {
    pub files: RefCell<Vec<Rc<FileMap>>>,
    expansions: RefCell<Vec<ExpnInfo>>,
    file_loader: Box<FileLoader>
}

impl CodeMap {
    pub fn new() -> CodeMap {
        CodeMap {
            files: RefCell::new(Vec::new()),
            expansions: RefCell::new(Vec::new()),
            file_loader: Box::new(RealFileLoader)
        }
    }

    pub fn with_file_loader(file_loader: Box<FileLoader>) -> CodeMap {
        CodeMap {
            files: RefCell::new(Vec::new()),
            expansions: RefCell::new(Vec::new()),
            file_loader: file_loader
        }
    }

    pub fn file_exists(&self, path: &Path) -> bool {
        self.file_loader.file_exists(path)
    }

    pub fn load_file(&self, path: &Path) -> io::Result<Rc<FileMap>> {
        let src = try!(self.file_loader.read_file(path));
        Ok(self.new_filemap(path.to_str().unwrap().to_string(), src))
    }

    fn next_start_pos(&self) -> usize {
        let files = self.files.borrow();
        match files.last() {
            None => 0,
            // Add one so there is some space between files. This lets us distinguish
            // positions in the codemap, even in the presence of zero-length files.
            Some(last) => last.end_pos.to_usize() + 1,
        }
    }

    /// Creates a new filemap without setting its line information. If you don't
    /// intend to set the line information yourself, you should use new_filemap_and_lines.
    pub fn new_filemap(&self, filename: FileName, mut src: String) -> Rc<FileMap> {
        let start_pos = self.next_start_pos();
        let mut files = self.files.borrow_mut();

        // Remove utf-8 BOM if any.
        if src.starts_with("\u{feff}") {
            src.drain(..3);
        }

        let end_pos = start_pos + src.len();

        let filemap = Rc::new(FileMap {
            name: filename,
            src: Some(Rc::new(src)),
            start_pos: Pos::from_usize(start_pos),
            end_pos: Pos::from_usize(end_pos),
            lines: RefCell::new(Vec::new()),
            multibyte_chars: RefCell::new(Vec::new()),
        });

        files.push(filemap.clone());

        filemap
    }

    /// Creates a new filemap and sets its line information.
    pub fn new_filemap_and_lines(&self, filename: &str, src: &str) -> Rc<FileMap> {
        let fm = self.new_filemap(filename.to_string(), src.to_owned());
        let mut byte_pos: u32 = 0;
        for line in src.lines() {
            // register the start of this line
            fm.next_line(BytePos(byte_pos));

            // update byte_pos to include this line and the \n at the end
            byte_pos += line.len() as u32 + 1;
        }
        fm
    }


    /// Allocates a new FileMap representing a source file from an external
    /// crate. The source code of such an "imported filemap" is not available,
    /// but we still know enough to generate accurate debuginfo location
    /// information for things inlined from other crates.
    pub fn new_imported_filemap(&self,
                                filename: FileName,
                                source_len: usize,
                                mut file_local_lines: Vec<BytePos>,
                                mut file_local_multibyte_chars: Vec<MultiByteChar>)
                                -> Rc<FileMap> {
        let start_pos = self.next_start_pos();
        let mut files = self.files.borrow_mut();

        let end_pos = Pos::from_usize(start_pos + source_len);
        let start_pos = Pos::from_usize(start_pos);

        for pos in &mut file_local_lines {
            *pos = *pos + start_pos;
        }

        for mbc in &mut file_local_multibyte_chars {
            mbc.pos = mbc.pos + start_pos;
        }

        let filemap = Rc::new(FileMap {
            name: filename,
            src: None,
            start_pos: start_pos,
            end_pos: end_pos,
            lines: RefCell::new(file_local_lines),
            multibyte_chars: RefCell::new(file_local_multibyte_chars),
        });

        files.push(filemap.clone());

        filemap
    }

    pub fn mk_substr_filename(&self, sp: Span) -> String {
        let pos = self.lookup_char_pos(sp.lo);
        (format!("<{}:{}:{}>",
                 pos.file.name,
                 pos.line,
                 pos.col.to_usize() + 1)).to_string()
    }

    /// Lookup source information about a BytePos
    pub fn lookup_char_pos(&self, pos: BytePos) -> Loc {
        let chpos = self.bytepos_to_file_charpos(pos);
        match self.lookup_line(pos) {
            Ok(FileMapAndLine { fm: f, line: a }) => {
                let line = a + 1; // Line numbers start at 1
                let linebpos = (*f.lines.borrow())[a];
                let linechpos = self.bytepos_to_file_charpos(linebpos);
                debug!("byte pos {:?} is on the line at byte pos {:?}",
                       pos, linebpos);
                debug!("char pos {:?} is on the line at char pos {:?}",
                       chpos, linechpos);
                debug!("byte is on line: {}", line);
                assert!(chpos >= linechpos);
                Loc {
                    file: f,
                    line: line,
                    col: chpos - linechpos,
                }
            }
            Err(f) => {
                Loc {
                    file: f,
                    line: 0,
                    col: chpos,
                }
            }
        }
    }

    // If the relevant filemap is empty, we don't return a line number.
    fn lookup_line(&self, pos: BytePos) -> Result<FileMapAndLine, Rc<FileMap>> {
        let idx = self.lookup_filemap_idx(pos);

        let files = self.files.borrow();
        let f = (*files)[idx].clone();

        let len = f.lines.borrow().len();
        if len == 0 {
            return Err(f);
        }

        let mut a = 0;
        {
            let lines = f.lines.borrow();
            let mut b = lines.len();
            while b - a > 1 {
                let m = (a + b) / 2;
                if (*lines)[m] > pos {
                    b = m;
                } else {
                    a = m;
                }
            }
            assert!(a <= lines.len());
        }
        Ok(FileMapAndLine { fm: f, line: a })
    }

    pub fn lookup_char_pos_adj(&self, pos: BytePos) -> LocWithOpt {
        let loc = self.lookup_char_pos(pos);
        LocWithOpt {
            filename: loc.file.name.to_string(),
            line: loc.line,
            col: loc.col,
            file: Some(loc.file)
        }
    }

    pub fn span_to_string(&self, sp: Span) -> String {
        if self.files.borrow().is_empty() && sp == DUMMY_SP {
            return "no-location".to_string();
        }

        let lo = self.lookup_char_pos_adj(sp.lo);
        let hi = self.lookup_char_pos_adj(sp.hi);
        return (format!("{}:{}:{}: {}:{}",
                        lo.filename,
                        lo.line,
                        lo.col.to_usize() + 1,
                        hi.line,
                        hi.col.to_usize() + 1)).to_string()
    }

    // Returns true if two spans have the same callee
    // (Assumes the same ExpnFormat implies same callee)
    fn match_callees(&self, sp_a: &Span, sp_b: &Span) -> bool {
        let fmt_a = self
            .with_expn_info(sp_a.expn_id,
                            |ei| ei.map(|ei| ei.callee.format.clone()));

        let fmt_b = self
            .with_expn_info(sp_b.expn_id,
                            |ei| ei.map(|ei| ei.callee.format.clone()));
        fmt_a == fmt_b
    }

    /// Returns a formatted string showing the expansion chain of a span
    ///
    /// Spans are printed in the following format:
    ///
    /// filename:start_line:col: end_line:col
    /// snippet
    ///   Callee:
    ///   Callee span
    ///   Callsite:
    ///   Callsite span
    ///
    /// Callees and callsites are printed recursively (if available, otherwise header
    /// and span is omitted), expanding into their own callee/callsite spans.
    /// Each layer of recursion has an increased indent, and snippets are truncated
    /// to at most 50 characters. Finally, recursive calls to the same macro are squashed,
    /// with '...' used to represent any number of recursive calls.
    pub fn span_to_expanded_string(&self, sp: Span) -> String {
        self.span_to_expanded_string_internal(sp, "")
    }

    fn span_to_expanded_string_internal(&self, sp:Span, indent: &str) -> String {
        let mut indent = indent.to_owned();
        let mut output = "".to_owned();
        let span_str = self.span_to_string(sp);
        let mut span_snip = self.span_to_snippet(sp)
            .unwrap_or("Snippet unavailable".to_owned());
        if span_snip.len() > 50 {
            span_snip.truncate(50);
            span_snip.push_str("...");
        }
        output.push_str(&format!("{}{}\n{}`{}`\n", indent, span_str, indent, span_snip));

        if sp.expn_id == NO_EXPANSION || sp.expn_id == COMMAND_LINE_EXPN {
            return output;
        }

        let mut callee = self.with_expn_info(sp.expn_id,
                                             |ei| ei.and_then(|ei| ei.callee.span.clone()));
        let mut callsite = self.with_expn_info(sp.expn_id,
                                               |ei| ei.map(|ei| ei.call_site.clone()));

        indent.push_str("  ");
        let mut is_recursive = false;

        while callee.is_some() && self.match_callees(&sp, &callee.unwrap()) {
            callee = self.with_expn_info(callee.unwrap().expn_id,
                                         |ei| ei.and_then(|ei| ei.callee.span.clone()));
            is_recursive = true;
        }
        if let Some(span) = callee {
            output.push_str(&indent);
            output.push_str("Callee:\n");
            if is_recursive {
                output.push_str(&indent);
                output.push_str("...\n");
            }
            output.push_str(&(self.span_to_expanded_string_internal(span, &indent)));
        }

        is_recursive = false;
        while callsite.is_some() && self.match_callees(&sp, &callsite.unwrap()) {
            callsite = self.with_expn_info(callsite.unwrap().expn_id,
                                           |ei| ei.map(|ei| ei.call_site.clone()));
            is_recursive = true;
        }
        if let Some(span) = callsite {
            output.push_str(&indent);
            output.push_str("Callsite:\n");
            if is_recursive {
                output.push_str(&indent);
                output.push_str("...\n");
            }
            output.push_str(&(self.span_to_expanded_string_internal(span, &indent)));
        }
        output
    }

    pub fn span_to_filename(&self, sp: Span) -> FileName {
        self.lookup_char_pos(sp.lo).file.name.to_string()
    }

    pub fn span_to_lines(&self, sp: Span) -> FileLinesResult {
        if sp.lo > sp.hi {
            return Err(SpanLinesError::IllFormedSpan(sp));
        }

        let lo = self.lookup_char_pos(sp.lo);
        let hi = self.lookup_char_pos(sp.hi);

        if lo.file.start_pos != hi.file.start_pos {
            return Err(SpanLinesError::DistinctSources(DistinctSources {
                begin: (lo.file.name.clone(), lo.file.start_pos),
                end: (hi.file.name.clone(), hi.file.start_pos),
            }));
        }
        assert!(hi.line >= lo.line);

        let mut lines = Vec::with_capacity(hi.line - lo.line + 1);

        // The span starts partway through the first line,
        // but after that it starts from offset 0.
        let mut start_col = lo.col;

        // For every line but the last, it extends from `start_col`
        // and to the end of the line. Be careful because the line
        // numbers in Loc are 1-based, so we subtract 1 to get 0-based
        // lines.
        for line_index in lo.line-1 .. hi.line-1 {
            let line_len = lo.file.get_line(line_index).map(|s| s.len()).unwrap_or(0);
            lines.push(LineInfo { line_index: line_index,
                                  start_col: start_col,
                                  end_col: CharPos::from_usize(line_len) });
            start_col = CharPos::from_usize(0);
        }

        // For the last line, it extends from `start_col` to `hi.col`:
        lines.push(LineInfo { line_index: hi.line - 1,
                              start_col: start_col,
                              end_col: hi.col });

        Ok(FileLines {file: lo.file, lines: lines})
    }

    pub fn span_to_snippet(&self, sp: Span) -> Result<String, SpanSnippetError> {
        if sp.lo > sp.hi {
            return Err(SpanSnippetError::IllFormedSpan(sp));
        }

        let local_begin = self.lookup_byte_offset(sp.lo);
        let local_end = self.lookup_byte_offset(sp.hi);

        if local_begin.fm.start_pos != local_end.fm.start_pos {
            return Err(SpanSnippetError::DistinctSources(DistinctSources {
                begin: (local_begin.fm.name.clone(),
                        local_begin.fm.start_pos),
                end: (local_end.fm.name.clone(),
                      local_end.fm.start_pos)
            }));
        } else {
            match local_begin.fm.src {
                Some(ref src) => {
                    let start_index = local_begin.pos.to_usize();
                    let end_index = local_end.pos.to_usize();
                    let source_len = (local_begin.fm.end_pos -
                                      local_begin.fm.start_pos).to_usize();

                    if start_index > end_index || end_index > source_len {
                        return Err(SpanSnippetError::MalformedForCodemap(
                            MalformedCodemapPositions {
                                name: local_begin.fm.name.clone(),
                                source_len: source_len,
                                begin_pos: local_begin.pos,
                                end_pos: local_end.pos,
                            }));
                    }

                    return Ok((&src[start_index..end_index]).to_string())
                }
                None => {
                    return Err(SpanSnippetError::SourceNotAvailable {
                        filename: local_begin.fm.name.clone()
                    });
                }
            }
        }
    }

    pub fn get_filemap(&self, filename: &str) -> Rc<FileMap> {
        for fm in self.files.borrow().iter() {
            if filename == fm.name {
                return fm.clone();
            }
        }
        panic!("asking for {} which we don't know about", filename);
    }

    /// For a global BytePos compute the local offset within the containing FileMap
    pub fn lookup_byte_offset(&self, bpos: BytePos) -> FileMapAndBytePos {
        let idx = self.lookup_filemap_idx(bpos);
        let fm = (*self.files.borrow())[idx].clone();
        let offset = bpos - fm.start_pos;
        FileMapAndBytePos {fm: fm, pos: offset}
    }

    /// Converts an absolute BytePos to a CharPos relative to the filemap.
    pub fn bytepos_to_file_charpos(&self, bpos: BytePos) -> CharPos {
        let idx = self.lookup_filemap_idx(bpos);
        let files = self.files.borrow();
        let map = &(*files)[idx];

        // The number of extra bytes due to multibyte chars in the FileMap
        let mut total_extra_bytes = 0;

        for mbc in map.multibyte_chars.borrow().iter() {
            debug!("{}-byte char at {:?}", mbc.bytes, mbc.pos);
            if mbc.pos < bpos {
                // every character is at least one byte, so we only
                // count the actual extra bytes.
                total_extra_bytes += mbc.bytes - 1;
                // We should never see a byte position in the middle of a
                // character
                assert!(bpos.to_usize() >= mbc.pos.to_usize() + mbc.bytes);
            } else {
                break;
            }
        }

        assert!(map.start_pos.to_usize() + total_extra_bytes <= bpos.to_usize());
        CharPos(bpos.to_usize() - map.start_pos.to_usize() - total_extra_bytes)
    }

    // Return the index of the filemap (in self.files) which contains pos.
    fn lookup_filemap_idx(&self, pos: BytePos) -> usize {
        let files = self.files.borrow();
        let files = &*files;
        let count = files.len();

        // Binary search for the filemap.
        let mut a = 0;
        let mut b = count;
        while b - a > 1 {
            let m = (a + b) / 2;
            if files[m].start_pos > pos {
                b = m;
            } else {
                a = m;
            }
        }

        assert!(a < count, "position {} does not resolve to a source location", pos.to_usize());

        return a;
    }

    pub fn record_expansion(&self, expn_info: ExpnInfo) -> ExpnId {
        let mut expansions = self.expansions.borrow_mut();
        expansions.push(expn_info);
        let len = expansions.len();
        if len > u32::max_value() as usize {
            panic!("too many ExpnInfo's!");
        }
        ExpnId(len as u32 - 1)
    }

    pub fn with_expn_info<T, F>(&self, id: ExpnId, f: F) -> T where
        F: FnOnce(Option<&ExpnInfo>) -> T,
    {
        match id {
            NO_EXPANSION | COMMAND_LINE_EXPN => f(None),
            ExpnId(i) => f(Some(&(*self.expansions.borrow())[i as usize]))
        }
    }

    /// Check if a span is "internal" to a macro in which #[unstable]
    /// items can be used (that is, a macro marked with
    /// `#[allow_internal_unstable]`).
    pub fn span_allows_unstable(&self, span: Span) -> bool {
        debug!("span_allows_unstable(span = {:?})", span);
        let mut allows_unstable = false;
        let mut expn_id = span.expn_id;
        loop {
            let quit = self.with_expn_info(expn_id, |expninfo| {
                debug!("span_allows_unstable: expninfo = {:?}", expninfo);
                expninfo.map_or(/* hit the top level */ true, |info| {

                    let span_comes_from_this_expansion =
                        info.callee.span.map_or(span == info.call_site, |mac_span| {
                            mac_span.contains(span)
                        });

                    debug!("span_allows_unstable: span: {:?} call_site: {:?} callee: {:?}",
                           (span.lo, span.hi),
                           (info.call_site.lo, info.call_site.hi),
                           info.callee.span.map(|x| (x.lo, x.hi)));
                    debug!("span_allows_unstable: from this expansion? {}, allows unstable? {}",
                           span_comes_from_this_expansion,
                           info.callee.allow_internal_unstable);
                    if span_comes_from_this_expansion {
                        allows_unstable = info.callee.allow_internal_unstable;
                        // we've found the right place, stop looking
                        true
                    } else {
                        // not the right place, keep looking
                        expn_id = info.call_site.expn_id;
                        false
                    }
                })
            });
            if quit {
                break
            }
        }
        debug!("span_allows_unstable? {}", allows_unstable);
        allows_unstable
    }

    pub fn count_lines(&self) -> usize {
        self.files.borrow().iter().fold(0, |a, f| a + f.count_lines())
    }
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
    begin: (String, BytePos),
    end: (String, BytePos)
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct MalformedCodemapPositions {
    name: String,
    source_len: usize,
    begin_pos: BytePos,
    end_pos: BytePos
}


// _____________________________________________________________________________
// Tests
//

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn t1 () {
        let cm = CodeMap::new();
        let fm = cm.new_filemap("blork.rs".to_string(),
                                "first line.\nsecond line".to_string());
        fm.next_line(BytePos(0));
        // Test we can get lines with partial line info.
        assert_eq!(fm.get_line(0), Some("first line."));
        // TESTING BROKEN BEHAVIOR: line break declared before actual line break.
        fm.next_line(BytePos(10));
        assert_eq!(fm.get_line(1), Some("."));
        fm.next_line(BytePos(12));
        assert_eq!(fm.get_line(2), Some("second line"));
    }

    #[test]
    #[should_panic]
    fn t2 () {
        let cm = CodeMap::new();
        let fm = cm.new_filemap("blork.rs".to_string(),
                                "first line.\nsecond line".to_string());
        // TESTING *REALLY* BROKEN BEHAVIOR:
        fm.next_line(BytePos(0));
        fm.next_line(BytePos(10));
        fm.next_line(BytePos(2));
    }

    fn init_code_map() -> CodeMap {
        let cm = CodeMap::new();
        let fm1 = cm.new_filemap("blork.rs".to_string(),
                                 "first line.\nsecond line".to_string());
        let fm2 = cm.new_filemap("empty.rs".to_string(),
                                 "".to_string());
        let fm3 = cm.new_filemap("blork2.rs".to_string(),
                                 "first line.\nsecond line".to_string());

        fm1.next_line(BytePos(0));
        fm1.next_line(BytePos(12));
        fm2.next_line(fm2.start_pos);
        fm3.next_line(fm3.start_pos);
        fm3.next_line(fm3.start_pos + BytePos(12));

        cm
    }

    #[test]
    fn t3() {
        // Test lookup_byte_offset
        let cm = init_code_map();

        let fmabp1 = cm.lookup_byte_offset(BytePos(23));
        assert_eq!(fmabp1.fm.name, "blork.rs");
        assert_eq!(fmabp1.pos, BytePos(23));

        let fmabp1 = cm.lookup_byte_offset(BytePos(24));
        assert_eq!(fmabp1.fm.name, "empty.rs");
        assert_eq!(fmabp1.pos, BytePos(0));

        let fmabp2 = cm.lookup_byte_offset(BytePos(25));
        assert_eq!(fmabp2.fm.name, "blork2.rs");
        assert_eq!(fmabp2.pos, BytePos(0));
    }

    #[test]
    fn t4() {
        // Test bytepos_to_file_charpos
        let cm = init_code_map();

        let cp1 = cm.bytepos_to_file_charpos(BytePos(22));
        assert_eq!(cp1, CharPos(22));

        let cp2 = cm.bytepos_to_file_charpos(BytePos(25));
        assert_eq!(cp2, CharPos(0));
    }

    #[test]
    fn t5() {
        // Test zero-length filemaps.
        let cm = init_code_map();

        let loc1 = cm.lookup_char_pos(BytePos(22));
        assert_eq!(loc1.file.name, "blork.rs");
        assert_eq!(loc1.line, 2);
        assert_eq!(loc1.col, CharPos(10));

        let loc2 = cm.lookup_char_pos(BytePos(25));
        assert_eq!(loc2.file.name, "blork2.rs");
        assert_eq!(loc2.line, 1);
        assert_eq!(loc2.col, CharPos(0));
    }

    fn init_code_map_mbc() -> CodeMap {
        let cm = CodeMap::new();
        // € is a three byte utf8 char.
        let fm1 =
            cm.new_filemap("blork.rs".to_string(),
                           "fir€st €€€€ line.\nsecond line".to_string());
        let fm2 = cm.new_filemap("blork2.rs".to_string(),
                                 "first line€€.\n€ second line".to_string());

        fm1.next_line(BytePos(0));
        fm1.next_line(BytePos(28));
        fm2.next_line(fm2.start_pos);
        fm2.next_line(fm2.start_pos + BytePos(20));

        fm1.record_multibyte_char(BytePos(3), 3);
        fm1.record_multibyte_char(BytePos(9), 3);
        fm1.record_multibyte_char(BytePos(12), 3);
        fm1.record_multibyte_char(BytePos(15), 3);
        fm1.record_multibyte_char(BytePos(18), 3);
        fm2.record_multibyte_char(fm2.start_pos + BytePos(10), 3);
        fm2.record_multibyte_char(fm2.start_pos + BytePos(13), 3);
        fm2.record_multibyte_char(fm2.start_pos + BytePos(18), 3);

        cm
    }

    #[test]
    fn t6() {
        // Test bytepos_to_file_charpos in the presence of multi-byte chars
        let cm = init_code_map_mbc();

        let cp1 = cm.bytepos_to_file_charpos(BytePos(3));
        assert_eq!(cp1, CharPos(3));

        let cp2 = cm.bytepos_to_file_charpos(BytePos(6));
        assert_eq!(cp2, CharPos(4));

        let cp3 = cm.bytepos_to_file_charpos(BytePos(56));
        assert_eq!(cp3, CharPos(12));

        let cp4 = cm.bytepos_to_file_charpos(BytePos(61));
        assert_eq!(cp4, CharPos(15));
    }

    #[test]
    fn t7() {
        // Test span_to_lines for a span ending at the end of filemap
        let cm = init_code_map();
        let span = Span {lo: BytePos(12), hi: BytePos(23), expn_id: NO_EXPANSION};
        let file_lines = cm.span_to_lines(span).unwrap();

        assert_eq!(file_lines.file.name, "blork.rs");
        assert_eq!(file_lines.lines.len(), 1);
        assert_eq!(file_lines.lines[0].line_index, 1);
    }

    /// Given a string like " ^~~~~~~~~~~~ ", produces a span
    /// coverting that range. The idea is that the string has the same
    /// length as the input, and we uncover the byte positions.  Note
    /// that this can span lines and so on.
    fn span_from_selection(input: &str, selection: &str) -> Span {
        assert_eq!(input.len(), selection.len());
        let left_index = selection.find('^').unwrap() as u32;
        let right_index = selection.rfind('~').unwrap() as u32;
        Span { lo: BytePos(left_index), hi: BytePos(right_index + 1), expn_id: NO_EXPANSION }
    }

    /// Test span_to_snippet and span_to_lines for a span coverting 3
    /// lines in the middle of a file.
    #[test]
    fn span_to_snippet_and_lines_spanning_multiple_lines() {
        let cm = CodeMap::new();
        let inputtext = "aaaaa\nbbbbBB\nCCC\nDDDDDddddd\neee\n";
        let selection = "     \n    ^~\n~~~\n~~~~~     \n   \n";
        cm.new_filemap_and_lines("blork.rs", inputtext);
        let span = span_from_selection(inputtext, selection);

        // check that we are extracting the text we thought we were extracting
        assert_eq!(&cm.span_to_snippet(span).unwrap(), "BB\nCCC\nDDDDD");

        // check that span_to_lines gives us the complete result with the lines/cols we expected
        let lines = cm.span_to_lines(span).unwrap();
        let expected = vec![
            LineInfo { line_index: 1, start_col: CharPos(4), end_col: CharPos(6) },
            LineInfo { line_index: 2, start_col: CharPos(0), end_col: CharPos(3) },
            LineInfo { line_index: 3, start_col: CharPos(0), end_col: CharPos(5) }
            ];
        assert_eq!(lines.lines, expected);
    }

    #[test]
    fn t8() {
        // Test span_to_snippet for a span ending at the end of filemap
        let cm = init_code_map();
        let span = Span {lo: BytePos(12), hi: BytePos(23), expn_id: NO_EXPANSION};
        let snippet = cm.span_to_snippet(span);

        assert_eq!(snippet, Ok("second line".to_string()));
    }

    #[test]
    fn t9() {
        // Test span_to_str for a span ending at the end of filemap
        let cm = init_code_map();
        let span = Span {lo: BytePos(12), hi: BytePos(23), expn_id: NO_EXPANSION};
        let sstr =  cm.span_to_string(span);

        assert_eq!(sstr, "blork.rs:2:1: 2:12");
    }

    #[test]
    fn t10() {
        // Test span_to_expanded_string works in base case (no expansion)
        let cm = init_code_map();
        let span = Span { lo: BytePos(0), hi: BytePos(11), expn_id: NO_EXPANSION };
        let sstr = cm.span_to_expanded_string(span);
        assert_eq!(sstr, "blork.rs:1:1: 1:12\n`first line.`\n");

        let span = Span { lo: BytePos(12), hi: BytePos(23), expn_id: NO_EXPANSION };
        let sstr =  cm.span_to_expanded_string(span);
        assert_eq!(sstr, "blork.rs:2:1: 2:12\n`second line`\n");
    }

    #[test]
    fn t11() {
        // Test span_to_expanded_string works with expansion
        use ast::Name;
        let cm = init_code_map();
        let root = Span { lo: BytePos(0), hi: BytePos(11), expn_id: NO_EXPANSION };
        let format = ExpnFormat::MacroBang(Name(0u32));
        let callee = NameAndSpan { format: format,
                                   allow_internal_unstable: false,
                                   span: None };

        let info = ExpnInfo { call_site: root, callee: callee };
        let id = cm.record_expansion(info);
        let sp = Span { lo: BytePos(12), hi: BytePos(23), expn_id: id };

        let sstr = cm.span_to_expanded_string(sp);
        assert_eq!(sstr,
                   "blork.rs:2:1: 2:12\n`second line`\n  Callsite:\n  \
                    blork.rs:1:1: 1:12\n  `first line.`\n");
    }

    fn init_expansion_chain(cm: &CodeMap) -> Span {
        // Creates an expansion chain containing two recursive calls
        // root -> expA -> expA -> expB -> expB -> end
        use ast::Name;

        let root = Span { lo: BytePos(0), hi: BytePos(11), expn_id: NO_EXPANSION };

        let format_root = ExpnFormat::MacroBang(Name(0u32));
        let callee_root = NameAndSpan { format: format_root,
                                        allow_internal_unstable: false,
                                        span: Some(root) };

        let info_a1 = ExpnInfo { call_site: root, callee: callee_root };
        let id_a1 = cm.record_expansion(info_a1);
        let span_a1 = Span { lo: BytePos(12), hi: BytePos(23), expn_id: id_a1 };

        let format_a = ExpnFormat::MacroBang(Name(1u32));
        let callee_a = NameAndSpan { format: format_a,
                                      allow_internal_unstable: false,
                                      span: Some(span_a1) };

        let info_a2 = ExpnInfo { call_site: span_a1, callee: callee_a.clone() };
        let id_a2 = cm.record_expansion(info_a2);
        let span_a2 = Span { lo: BytePos(12), hi: BytePos(23), expn_id: id_a2 };

        let info_b1 = ExpnInfo { call_site: span_a2, callee: callee_a };
        let id_b1 = cm.record_expansion(info_b1);
        let span_b1 = Span { lo: BytePos(25), hi: BytePos(36), expn_id: id_b1 };

        let format_b = ExpnFormat::MacroBang(Name(2u32));
        let callee_b = NameAndSpan { format: format_b,
                                     allow_internal_unstable: false,
                                     span: None };

        let info_b2 = ExpnInfo { call_site: span_b1, callee: callee_b.clone() };
        let id_b2 = cm.record_expansion(info_b2);
        let span_b2 = Span { lo: BytePos(25), hi: BytePos(36), expn_id: id_b2 };

        let info_end = ExpnInfo { call_site: span_b2, callee: callee_b };
        let id_end = cm.record_expansion(info_end);
        Span { lo: BytePos(37), hi: BytePos(48), expn_id: id_end }
    }

    #[test]
    fn t12() {
        // Test span_to_expanded_string collapses recursive macros and handles
        // recursive callsite and callee expansions
        let cm = init_code_map();
        let end = init_expansion_chain(&cm);
        let sstr = cm.span_to_expanded_string(end);
        let res_str =
r"blork2.rs:2:1: 2:12
`second line`
  Callsite:
  ...
  blork2.rs:1:1: 1:12
  `first line.`
    Callee:
    blork.rs:2:1: 2:12
    `second line`
      Callee:
      blork.rs:1:1: 1:12
      `first line.`
      Callsite:
      blork.rs:1:1: 1:12
      `first line.`
    Callsite:
    ...
    blork.rs:2:1: 2:12
    `second line`
      Callee:
      blork.rs:1:1: 1:12
      `first line.`
      Callsite:
      blork.rs:1:1: 1:12
      `first line.`
";
        assert_eq!(sstr, res_str);
    }
}
