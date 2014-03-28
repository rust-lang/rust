// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!

The CodeMap tracks all the source code used within a single crate, mapping
from integer byte positions to the original source code location. Each bit of
source parsed during crate parsing (typically files, in-memory strings, or
various bits of macro expansion) cover a continuous range of bytes in the
CodeMap and are represented by FileMaps. Byte positions are stored in `spans`
and used pervasively in the compiler. They are absolute positions within the
CodeMap, which upon request can be converted to line and column information,
source code snippets, etc.

*/

use std::cell::RefCell;
use std::rc::Rc;
use serialize::{Encodable, Decodable, Encoder, Decoder};

pub trait Pos {
    fn from_uint(n: uint) -> Self;
    fn to_uint(&self) -> uint;
}

/// A byte offset. Keep this small (currently 32-bits), as AST contains
/// a lot of them.
#[deriving(Clone, Eq, TotalEq, Hash, Ord, Show)]
pub struct BytePos(u32);

/// A character offset. Because of multibyte utf8 characters, a byte offset
/// is not equivalent to a character offset. The CodeMap will convert BytePos
/// values to CharPos values as necessary.
#[deriving(Eq, Hash, Ord, Show)]
pub struct CharPos(uint);

// FIXME: Lots of boilerplate in these impls, but so far my attempts to fix
// have been unsuccessful

impl Pos for BytePos {
    fn from_uint(n: uint) -> BytePos { BytePos(n as u32) }
    fn to_uint(&self) -> uint { let BytePos(n) = *self; n as uint }
}

impl Add<BytePos, BytePos> for BytePos {
    fn add(&self, rhs: &BytePos) -> BytePos {
        BytePos((self.to_uint() + rhs.to_uint()) as u32)
    }
}

impl Sub<BytePos, BytePos> for BytePos {
    fn sub(&self, rhs: &BytePos) -> BytePos {
        BytePos((self.to_uint() - rhs.to_uint()) as u32)
    }
}

impl Pos for CharPos {
    fn from_uint(n: uint) -> CharPos { CharPos(n) }
    fn to_uint(&self) -> uint { let CharPos(n) = *self; n }
}

impl Add<CharPos,CharPos> for CharPos {
    fn add(&self, rhs: &CharPos) -> CharPos {
        CharPos(self.to_uint() + rhs.to_uint())
    }
}

impl Sub<CharPos,CharPos> for CharPos {
    fn sub(&self, rhs: &CharPos) -> CharPos {
        CharPos(self.to_uint() - rhs.to_uint())
    }
}

/**
Spans represent a region of code, used for error reporting. Positions in spans
are *absolute* positions from the beginning of the codemap, not positions
relative to FileMaps. Methods on the CodeMap can be used to relate spans back
to the original source.
*/
#[deriving(Clone, Show, Hash)]
pub struct Span {
    lo: BytePos,
    hi: BytePos,
    /// Information about where the macro came from, if this piece of
    /// code was created by a macro expansion.
    expn_info: Option<@ExpnInfo>
}

pub static DUMMY_SP: Span = Span { lo: BytePos(0), hi: BytePos(0), expn_info: None };

#[deriving(Clone, Eq, TotalEq, Encodable, Decodable, Hash)]
pub struct Spanned<T> {
    node: T,
    span: Span,
}

impl Eq for Span {
    fn eq(&self, other: &Span) -> bool {
        return (*self).lo == (*other).lo && (*self).hi == (*other).hi;
    }
    fn ne(&self, other: &Span) -> bool { !(*self).eq(other) }
}

impl TotalEq for Span {}

// FIXME: remove stage0 Encodables/Decodables after snapshot
#[cfg(stage0)]
impl<S:Encoder> Encodable<S> for Span {
    /* Note #1972 -- spans are encoded but not decoded */
    fn encode(&self, s: &mut S) {
        s.emit_nil()
    }
}

#[cfg(stage0)]
impl<D:Decoder> Decodable<D> for Span {
    fn decode(_d: &mut D) -> Span {
        DUMMY_SP
    }
}

#[cfg(not(stage0))]
impl<S:Encoder<E>, E> Encodable<S, E> for Span {
    /* Note #1972 -- spans are encoded but not decoded */
    fn encode(&self, s: &mut S) -> Result<(), E> {
        s.emit_nil()
    }
}

#[cfg(not(stage0))]
impl<D:Decoder<E>, E> Decodable<D, E> for Span {
    fn decode(_d: &mut D) -> Result<Span, E> {
        Ok(DUMMY_SP)
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
    Span {lo: lo, hi: hi, expn_info: None}
}

/// A source code location used for error reporting
pub struct Loc {
    /// Information about the original source
    file: Rc<FileMap>,
    /// The (1-based) line number
    line: uint,
    /// The (0-based) column offset
    col: CharPos
}

/// A source code location used as the result of lookup_char_pos_adj
// Actually, *none* of the clients use the filename *or* file field;
// perhaps they should just be removed.
pub struct LocWithOpt {
    filename: FileName,
    line: uint,
    col: CharPos,
    file: Option<Rc<FileMap>>,
}

// used to be structural records. Better names, anyone?
pub struct FileMapAndLine {fm: Rc<FileMap>, line: uint}
pub struct FileMapAndBytePos {fm: Rc<FileMap>, pos: BytePos}

/// The syntax with which a macro was invoked.
#[deriving(Clone, Hash, Show)]
pub enum MacroFormat {
    /// e.g. #[deriving(...)] <item>
    MacroAttribute,
    /// e.g. `format!()`
    MacroBang
}

#[deriving(Clone, Hash, Show)]
pub struct NameAndSpan {
    /// The name of the macro that was invoked to create the thing
    /// with this Span.
    name: ~str,
    /// The format with which the macro was invoked.
    format: MacroFormat,
    /// The span of the macro definition itself. The macro may not
    /// have a sensible definition span (e.g. something defined
    /// completely inside libsyntax) in which case this is None.
    span: Option<Span>
}

/// Extra information for tracking macro expansion of spans
#[deriving(Hash, Show)]
pub struct ExpnInfo {
    /// The location of the actual macro invocation, e.g. `let x =
    /// foo!();`
    ///
    /// This may recursively refer to other macro invocations, e.g. if
    /// `foo!()` invoked `bar!()` internally, and there was an
    /// expression inside `bar!`; the call_site of the expression in
    /// the expansion would point to the `bar!` invocation; that
    /// call_site span would have its own ExpnInfo, with the call_site
    /// pointing to the `foo!` invocation.
    call_site: Span,
    /// Information about the macro and its definition.
    ///
    /// The `callee` of the inner expression in the `call_site`
    /// example would point to the `macro_rules! bar { ... }` and that
    /// of the `bar!()` invocation would point to the `macro_rules!
    /// foo { ... }`.
    callee: NameAndSpan
}

pub type FileName = ~str;

pub struct FileLines {
    file: Rc<FileMap>,
    lines: Vec<uint>
}

/// Identifies an offset of a multi-byte character in a FileMap
pub struct MultiByteChar {
    /// The absolute offset of the character in the CodeMap
    pos: BytePos,
    /// The number of bytes, >=2
    bytes: uint,
}

/// A single source in the CodeMap
pub struct FileMap {
    /// The name of the file that the source came from, source that doesn't
    /// originate from files has names between angle brackets by convention,
    /// e.g. `<anon>`
    name: FileName,
    /// The complete source code
    src: ~str,
    /// The start position of this source in the CodeMap
    start_pos: BytePos,
    /// Locations of lines beginnings in the source code
    lines: RefCell<Vec<BytePos> >,
    /// Locations of multi-byte characters in the source code
    multibyte_chars: RefCell<Vec<MultiByteChar> >,
}

impl FileMap {
    // EFFECT: register a start-of-line offset in the
    // table of line-beginnings.
    // UNCHECKED INVARIANT: these offsets must be added in the right
    // order and must be in the right places; there is shared knowledge
    // about what ends a line between this file and parse.rs
    // WARNING: pos param here is the offset relative to start of CodeMap,
    // and CodeMap will append a newline when adding a filemap without a newline at the end,
    // so the safe way to call this is with value calculated as
    // filemap.start_pos + newline_offset_relative_to_the_start_of_filemap.
    pub fn next_line(&self, pos: BytePos) {
        // the new charpos must be > the last one (or it's the first one).
        let mut lines = self.lines.borrow_mut();;
        let line_len = lines.len();
        assert!(line_len == 0 || (*lines.get(line_len - 1) < pos))
        lines.push(pos);
    }

    // get a line from the list of pre-computed line-beginnings
    pub fn get_line(&self, line: int) -> ~str {
        let mut lines = self.lines.borrow_mut();
        let begin: BytePos = *lines.get(line as uint) - self.start_pos;
        let begin = begin.to_uint();
        let slice = self.src.slice_from(begin);
        match slice.find('\n') {
            Some(e) => slice.slice_to(e).to_owned(),
            None => slice.to_owned()
        }
    }

    pub fn record_multibyte_char(&self, pos: BytePos, bytes: uint) {
        assert!(bytes >=2 && bytes <= 4);
        let mbc = MultiByteChar {
            pos: pos,
            bytes: bytes,
        };
        self.multibyte_chars.borrow_mut().push(mbc);
    }

    pub fn is_real_file(&self) -> bool {
        !(self.name.starts_with("<") && self.name.ends_with(">"))
    }
}

pub struct CodeMap {
    files: RefCell<Vec<Rc<FileMap>>>
}

impl CodeMap {
    pub fn new() -> CodeMap {
        CodeMap {
            files: RefCell::new(Vec::new()),
        }
    }

    pub fn new_filemap(&self, filename: FileName, src: ~str) -> Rc<FileMap> {
        let mut files = self.files.borrow_mut();
        let start_pos = match files.last() {
            None => 0,
            Some(last) => last.start_pos.to_uint() + last.src.len(),
        };

        // Remove utf-8 BOM if any.
        // FIXME #12884: no efficient/safe way to remove from the start of a string
        // and reuse the allocation.
        let mut src = if src.starts_with("\ufeff") {
            src.as_slice().slice_from(3).into_owned()
        } else {
            src
        };

        // Append '\n' in case it's not already there.
        // This is a workaround to prevent CodeMap.lookup_filemap_idx from accidentally
        // overflowing into the next filemap in case the last byte of span is also the last
        // byte of filemap, which leads to incorrect results from CodeMap.span_to_*.
        if src.len() > 0 && !src.ends_with("\n") {
            src.push_char('\n');
        }

        let filemap = Rc::new(FileMap {
            name: filename,
            src: src,
            start_pos: Pos::from_uint(start_pos),
            lines: RefCell::new(Vec::new()),
            multibyte_chars: RefCell::new(Vec::new()),
        });

        files.push(filemap.clone());

        filemap
    }

    pub fn mk_substr_filename(&self, sp: Span) -> ~str {
        let pos = self.lookup_char_pos(sp.lo);
        format!("<{}:{}:{}>", pos.file.name, pos.line, pos.col.to_uint() + 1)
    }

    /// Lookup source information about a BytePos
    pub fn lookup_char_pos(&self, pos: BytePos) -> Loc {
        self.lookup_pos(pos)
    }

    pub fn lookup_char_pos_adj(&self, pos: BytePos) -> LocWithOpt {
        let loc = self.lookup_char_pos(pos);
        LocWithOpt {
            filename: loc.file.name.to_str(),
            line: loc.line,
            col: loc.col,
            file: Some(loc.file)
        }
    }

    pub fn span_to_str(&self, sp: Span) -> ~str {
        if self.files.borrow().len() == 0 && sp == DUMMY_SP {
            return ~"no-location";
        }

        let lo = self.lookup_char_pos_adj(sp.lo);
        let hi = self.lookup_char_pos_adj(sp.hi);
        return format!("{}:{}:{}: {}:{}", lo.filename,
                       lo.line, lo.col.to_uint() + 1, hi.line, hi.col.to_uint() + 1)
    }

    pub fn span_to_filename(&self, sp: Span) -> FileName {
        self.lookup_char_pos(sp.lo).file.name.to_str()
    }

    pub fn span_to_lines(&self, sp: Span) -> FileLines {
        let lo = self.lookup_char_pos(sp.lo);
        let hi = self.lookup_char_pos(sp.hi);
        let mut lines = Vec::new();
        for i in range(lo.line - 1u, hi.line as uint) {
            lines.push(i);
        };
        FileLines {file: lo.file, lines: lines}
    }

    pub fn span_to_snippet(&self, sp: Span) -> Option<~str> {
        let begin = self.lookup_byte_offset(sp.lo);
        let end = self.lookup_byte_offset(sp.hi);

        // FIXME #8256: this used to be an assert but whatever precondition
        // it's testing isn't true for all spans in the AST, so to allow the
        // caller to not have to fail (and it can't catch it since the CodeMap
        // isn't sendable), return None
        if begin.fm.start_pos != end.fm.start_pos {
            None
        } else {
            Some(begin.fm.src.slice( begin.pos.to_uint(), end.pos.to_uint()).to_owned())
        }
    }

    pub fn get_filemap(&self, filename: &str) -> Rc<FileMap> {
        for fm in self.files.borrow().iter() {
            if filename == fm.name {
                return fm.clone();
            }
        }
        fail!("asking for {} which we don't know about", filename);
    }

    fn lookup_filemap_idx(&self, pos: BytePos) -> uint {
        let files = self.files.borrow();
        let files = files;
        let len = files.len();
        let mut a = 0u;
        let mut b = len;
        while b - a > 1u {
            let m = (a + b) / 2u;
            if files.get(m).start_pos > pos {
                b = m;
            } else {
                a = m;
            }
        }
        // There can be filemaps with length 0. These have the same start_pos as the previous
        // filemap, but are not the filemaps we want (because they are length 0, they cannot
        // contain what we are looking for). So, rewind until we find a useful filemap.
        loop {
            let lines = files.get(a).lines.borrow();
            let lines = lines;
            if lines.len() > 0 {
                break;
            }
            if a == 0 {
                fail!("position {} does not resolve to a source location", pos.to_uint());
            }
            a -= 1;
        }
        if a >= len {
            fail!("position {} does not resolve to a source location", pos.to_uint())
        }

        return a;
    }

    fn lookup_line(&self, pos: BytePos) -> FileMapAndLine {
        let idx = self.lookup_filemap_idx(pos);

        let files = self.files.borrow();
        let f = files.get(idx).clone();
        let mut a = 0u;
        {
            let mut lines = f.lines.borrow_mut();
            let mut b = lines.len();
            while b - a > 1u {
                let m = (a + b) / 2u;
                if *lines.get(m) > pos { b = m; } else { a = m; }
            }
        }
        FileMapAndLine {fm: f, line: a}
    }

    fn lookup_pos(&self, pos: BytePos) -> Loc {
        let FileMapAndLine {fm: f, line: a} = self.lookup_line(pos);
        let line = a + 1u; // Line numbers start at 1
        let chpos = self.bytepos_to_file_charpos(pos);
        let linebpos = *f.lines.borrow().get(a);
        let linechpos = self.bytepos_to_file_charpos(linebpos);
        debug!("codemap: byte pos {:?} is on the line at byte pos {:?}",
               pos, linebpos);
        debug!("codemap: char pos {:?} is on the line at char pos {:?}",
               chpos, linechpos);
        debug!("codemap: byte is on line: {:?}", line);
        assert!(chpos >= linechpos);
        Loc {
            file: f,
            line: line,
            col: chpos - linechpos
        }
    }

    fn lookup_byte_offset(&self, bpos: BytePos) -> FileMapAndBytePos {
        let idx = self.lookup_filemap_idx(bpos);
        let fm = self.files.borrow().get(idx).clone();
        let offset = bpos - fm.start_pos;
        FileMapAndBytePos {fm: fm, pos: offset}
    }

    // Converts an absolute BytePos to a CharPos relative to the filemap.
    fn bytepos_to_file_charpos(&self, bpos: BytePos) -> CharPos {
        debug!("codemap: converting {:?} to char pos", bpos);
        let idx = self.lookup_filemap_idx(bpos);
        let files = self.files.borrow();
        let map = files.get(idx);

        // The number of extra bytes due to multibyte chars in the FileMap
        let mut total_extra_bytes = 0;

        for mbc in map.multibyte_chars.borrow().iter() {
            debug!("codemap: {:?}-byte char at {:?}", mbc.bytes, mbc.pos);
            if mbc.pos < bpos {
                // every character is at least one byte, so we only
                // count the actual extra bytes.
                total_extra_bytes += mbc.bytes - 1;
                // We should never see a byte position in the middle of a
                // character
                assert!(bpos.to_uint() >= mbc.pos.to_uint() + mbc.bytes);
            } else {
                break;
            }
        }

        assert!(map.start_pos.to_uint() + total_extra_bytes <= bpos.to_uint());
        CharPos(bpos.to_uint() - map.start_pos.to_uint() - total_extra_bytes)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn t1 () {
        let cm = CodeMap::new();
        let fm = cm.new_filemap(~"blork.rs",~"first line.\nsecond line");
        fm.next_line(BytePos(0));
        assert_eq!(&fm.get_line(0),&~"first line.");
        // TESTING BROKEN BEHAVIOR:
        fm.next_line(BytePos(10));
        assert_eq!(&fm.get_line(1),&~".");
    }

    #[test]
    #[should_fail]
    fn t2 () {
        let cm = CodeMap::new();
        let fm = cm.new_filemap(~"blork.rs",~"first line.\nsecond line");
        // TESTING *REALLY* BROKEN BEHAVIOR:
        fm.next_line(BytePos(0));
        fm.next_line(BytePos(10));
        fm.next_line(BytePos(2));
    }

    fn init_code_map() -> CodeMap {
        let cm = CodeMap::new();
        let fm1 = cm.new_filemap(~"blork.rs",~"first line.\nsecond line");
        let fm2 = cm.new_filemap(~"empty.rs",~"");
        let fm3 = cm.new_filemap(~"blork2.rs",~"first line.\nsecond line");

        fm1.next_line(BytePos(0));
        fm1.next_line(BytePos(12));
        fm2.next_line(BytePos(24));
        fm3.next_line(BytePos(24));
        fm3.next_line(BytePos(34));

        cm
    }

    #[test]
    fn t3() {
        // Test lookup_byte_offset
        let cm = init_code_map();

        let fmabp1 = cm.lookup_byte_offset(BytePos(22));
        assert_eq!(fmabp1.fm.name, ~"blork.rs");
        assert_eq!(fmabp1.pos, BytePos(22));

        let fmabp2 = cm.lookup_byte_offset(BytePos(24));
        assert_eq!(fmabp2.fm.name, ~"blork2.rs");
        assert_eq!(fmabp2.pos, BytePos(0));
    }

    #[test]
    fn t4() {
        // Test bytepos_to_file_charpos
        let cm = init_code_map();

        let cp1 = cm.bytepos_to_file_charpos(BytePos(22));
        assert_eq!(cp1, CharPos(22));

        let cp2 = cm.bytepos_to_file_charpos(BytePos(24));
        assert_eq!(cp2, CharPos(0));
    }

    #[test]
    fn t5() {
        // Test zero-length filemaps.
        let cm = init_code_map();

        let loc1 = cm.lookup_char_pos(BytePos(22));
        assert_eq!(loc1.file.name, ~"blork.rs");
        assert_eq!(loc1.line, 2);
        assert_eq!(loc1.col, CharPos(10));

        let loc2 = cm.lookup_char_pos(BytePos(24));
        assert_eq!(loc2.file.name, ~"blork2.rs");
        assert_eq!(loc2.line, 1);
        assert_eq!(loc2.col, CharPos(0));
    }

    fn init_code_map_mbc() -> CodeMap {
        let cm = CodeMap::new();
        // € is a three byte utf8 char.
        let fm1 = cm.new_filemap(~"blork.rs",~"fir€st €€€€ line.\nsecond line");
        let fm2 = cm.new_filemap(~"blork2.rs",~"first line€€.\n€ second line");

        fm1.next_line(BytePos(0));
        fm1.next_line(BytePos(22));
        fm2.next_line(BytePos(40));
        fm2.next_line(BytePos(58));

        fm1.record_multibyte_char(BytePos(3), 3);
        fm1.record_multibyte_char(BytePos(9), 3);
        fm1.record_multibyte_char(BytePos(12), 3);
        fm1.record_multibyte_char(BytePos(15), 3);
        fm1.record_multibyte_char(BytePos(18), 3);
        fm2.record_multibyte_char(BytePos(50), 3);
        fm2.record_multibyte_char(BytePos(53), 3);
        fm2.record_multibyte_char(BytePos(58), 3);

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
        let span = Span {lo: BytePos(12), hi: BytePos(23), expn_info: None};
        let file_lines = cm.span_to_lines(span);

        assert_eq!(file_lines.file.name, ~"blork.rs");
        assert_eq!(file_lines.lines.len(), 1);
        assert_eq!(*file_lines.lines.get(0), 1u);
    }

    #[test]
    fn t8() {
        // Test span_to_snippet for a span ending at the end of filemap
        let cm = init_code_map();
        let span = Span {lo: BytePos(12), hi: BytePos(23), expn_info: None};
        let snippet = cm.span_to_snippet(span);

        assert_eq!(snippet, Some(~"second line"));
    }

    #[test]
    fn t9() {
        // Test span_to_str for a span ending at the end of filemap
        let cm = init_code_map();
        let span = Span {lo: BytePos(12), hi: BytePos(23), expn_info: None};
        let sstr =  cm.span_to_str(span);

        assert_eq!(sstr, ~"blork.rs:2:1: 2:12");
    }
}
