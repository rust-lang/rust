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
use std::cmp;
use serialize::{Encodable, Decodable, Encoder, Decoder};

pub trait Pos {
    fn from_uint(n: uint) -> Self;
    fn to_uint(&self) -> uint;
}

/// A byte offset. Keep this small (currently 32-bits), as AST contains
/// a lot of them.
#[deriving(Clone, Eq, Hash, Ord, Show)]
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
    expn_info: Option<@ExpnInfo>
}

pub static DUMMY_SP: Span = Span { lo: BytePos(0), hi: BytePos(0), expn_info: None };

#[deriving(Clone, Eq, Encodable, Decodable, Hash)]
pub struct Spanned<T> {
    node: T,
    span: Span,
}

impl cmp::Eq for Span {
    fn eq(&self, other: &Span) -> bool {
        return (*self).lo == (*other).lo && (*self).hi == (*other).hi;
    }
    fn ne(&self, other: &Span) -> bool { !(*self).eq(other) }
}

impl<S:Encoder> Encodable<S> for Span {
    /* Note #1972 -- spans are encoded but not decoded */
    fn encode(&self, s: &mut S) {
        s.emit_nil()
    }
}

impl<D:Decoder> Decodable<D> for Span {
    fn decode(_d: &mut D) -> Span {
        DUMMY_SP
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
    file: @FileMap,
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
    file: Option<@FileMap>,
}

// used to be structural records. Better names, anyone?
pub struct FileMapAndLine {fm: @FileMap, line: uint}
pub struct FileMapAndBytePos {fm: @FileMap, pos: BytePos}

#[deriving(Clone, Hash, Show)]
pub enum MacroFormat {
    // e.g. #[deriving(...)] <item>
    MacroAttribute,
    // e.g. `format!()`
    MacroBang
}

#[deriving(Clone, Hash, Show)]
pub struct NameAndSpan {
    name: ~str,
    // the format with which the macro was invoked.
    format: MacroFormat,
    span: Option<Span>
}

/// Extra information for tracking macro expansion of spans
#[deriving(Hash, Show)]
pub struct ExpnInfo {
    call_site: Span,
    callee: NameAndSpan
}

pub type FileName = ~str;

pub struct FileLines
{
    file: @FileMap,
    lines: ~[uint]
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
    lines: RefCell<~[BytePos]>,
    /// Locations of multi-byte characters in the source code
    multibyte_chars: RefCell<~[MultiByteChar]>,
}

impl FileMap {
    // EFFECT: register a start-of-line offset in the
    // table of line-beginnings.
    // UNCHECKED INVARIANT: these offsets must be added in the right
    // order and must be in the right places; there is shared knowledge
    // about what ends a line between this file and parse.rs
    pub fn next_line(&self, pos: BytePos) {
        // the new charpos must be > the last one (or it's the first one).
        let mut lines = self.lines.borrow_mut();;
        let line_len = lines.get().len();
        assert!(line_len == 0 || (lines.get()[line_len - 1] < pos))
        lines.get().push(pos);
    }

    // get a line from the list of pre-computed line-beginnings
    pub fn get_line(&self, line: int) -> ~str {
        let mut lines = self.lines.borrow_mut();
        let begin: BytePos = lines.get()[line] - self.start_pos;
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
        let mut multibyte_chars = self.multibyte_chars.borrow_mut();
        multibyte_chars.get().push(mbc);
    }

    pub fn is_real_file(&self) -> bool {
        !(self.name.starts_with("<") && self.name.ends_with(">"))
    }
}

pub struct CodeMap {
    files: RefCell<~[@FileMap]>
}

impl CodeMap {
    pub fn new() -> CodeMap {
        CodeMap {
            files: RefCell::new(~[]),
        }
    }

    pub fn new_filemap(&self, filename: FileName, src: ~str) -> @FileMap {
        let mut files = self.files.borrow_mut();
        let start_pos = match files.get().last() {
            None => 0,
            Some(last) => last.start_pos.to_uint() + last.src.len(),
        };

        let filemap = @FileMap {
            name: filename,
            src: src,
            start_pos: Pos::from_uint(start_pos),
            lines: RefCell::new(~[]),
            multibyte_chars: RefCell::new(~[]),
        };

        files.get().push(filemap);

        return filemap;
    }

    pub fn mk_substr_filename(&self, sp: Span) -> ~str {
        let pos = self.lookup_char_pos(sp.lo);
        return format!("<{}:{}:{}>", pos.file.name,
                       pos.line, pos.col.to_uint() + 1)
    }

    /// Lookup source information about a BytePos
    pub fn lookup_char_pos(&self, pos: BytePos) -> Loc {
        return self.lookup_pos(pos);
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
        {
            let files = self.files.borrow();
            if files.get().len() == 0 && sp == DUMMY_SP {
                return ~"no-location";
            }
        }

        let lo = self.lookup_char_pos_adj(sp.lo);
        let hi = self.lookup_char_pos_adj(sp.hi);
        return format!("{}:{}:{}: {}:{}", lo.filename,
                       lo.line, lo.col.to_uint() + 1, hi.line, hi.col.to_uint() + 1)
    }

    pub fn span_to_filename(&self, sp: Span) -> FileName {
        let lo = self.lookup_char_pos(sp.lo);
        lo.file.name.to_str()
    }

    pub fn span_to_lines(&self, sp: Span) -> @FileLines {
        let lo = self.lookup_char_pos(sp.lo);
        let hi = self.lookup_char_pos(sp.hi);
        let mut lines = ~[];
        for i in range(lo.line - 1u, hi.line as uint) {
            lines.push(i);
        };
        return @FileLines {file: lo.file, lines: lines};
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

    pub fn get_filemap(&self, filename: &str) -> @FileMap {
        let files = self.files.borrow();
        for fm in files.get().iter() {
            if filename == fm.name {
                return *fm
            }
        }
        //XXjdm the following triggers a mismatched type bug
        //      (or expected function, found _|_)
        fail!(); // ("asking for " + filename + " which we don't know about");
    }
}

impl CodeMap {
    fn lookup_filemap_idx(&self, pos: BytePos) -> uint {
        let files = self.files.borrow();
        let files = files.get();
        let len = files.len();
        let mut a = 0u;
        let mut b = len;
        while b - a > 1u {
            let m = (a + b) / 2u;
            if files[m].start_pos > pos {
                b = m;
            } else {
                a = m;
            }
        }
        // There can be filemaps with length 0. These have the same start_pos as the previous
        // filemap, but are not the filemaps we want (because they are length 0, they cannot
        // contain what we are looking for). So, rewind until we find a useful filemap.
        loop {
            let lines = files[a].lines.borrow();
            let lines = lines.get();
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

    fn lookup_line(&self, pos: BytePos) -> FileMapAndLine
    {
        let idx = self.lookup_filemap_idx(pos);

        let files = self.files.borrow();
        let f = files.get()[idx];
        let mut a = 0u;
        let mut lines = f.lines.borrow_mut();
        let mut b = lines.get().len();
        while b - a > 1u {
            let m = (a + b) / 2u;
            if lines.get()[m] > pos { b = m; } else { a = m; }
        }
        return FileMapAndLine {fm: f, line: a};
    }

    fn lookup_pos(&self, pos: BytePos) -> Loc {
        let FileMapAndLine {fm: f, line: a} = self.lookup_line(pos);
        let line = a + 1u; // Line numbers start at 1
        let chpos = self.bytepos_to_file_charpos(pos);
        let lines = f.lines.borrow();
        let linebpos = lines.get()[a];
        let linechpos = self.bytepos_to_file_charpos(linebpos);
        debug!("codemap: byte pos {:?} is on the line at byte pos {:?}",
               pos, linebpos);
        debug!("codemap: char pos {:?} is on the line at char pos {:?}",
               chpos, linechpos);
        debug!("codemap: byte is on line: {:?}", line);
        assert!(chpos >= linechpos);
        return Loc {
            file: f,
            line: line,
            col: chpos - linechpos
        };
    }

    fn lookup_byte_offset(&self, bpos: BytePos)
        -> FileMapAndBytePos {
        let idx = self.lookup_filemap_idx(bpos);
        let files = self.files.borrow();
        let fm = files.get()[idx];
        let offset = bpos - fm.start_pos;
        return FileMapAndBytePos {fm: fm, pos: offset};
    }

    // Converts an absolute BytePos to a CharPos relative to the filemap.
    fn bytepos_to_file_charpos(&self, bpos: BytePos) -> CharPos {
        debug!("codemap: converting {:?} to char pos", bpos);
        let idx = self.lookup_filemap_idx(bpos);
        let files = self.files.borrow();
        let map = files.get()[idx];

        // The number of extra bytes due to multibyte chars in the FileMap
        let mut total_extra_bytes = 0;

        let multibyte_chars = map.multibyte_chars.borrow();
        for mbc in multibyte_chars.get().iter() {
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
        fm2.next_line(BytePos(23));
        fm3.next_line(BytePos(23));
        fm3.next_line(BytePos(33));

        cm
    }

    #[test]
    fn t3() {
        // Test lookup_byte_offset
        let cm = init_code_map();

        let fmabp1 = cm.lookup_byte_offset(BytePos(22));
        assert_eq!(fmabp1.fm.name, ~"blork.rs");
        assert_eq!(fmabp1.pos, BytePos(22));

        let fmabp2 = cm.lookup_byte_offset(BytePos(23));
        assert_eq!(fmabp2.fm.name, ~"blork2.rs");
        assert_eq!(fmabp2.pos, BytePos(0));
    }

    #[test]
    fn t4() {
        // Test bytepos_to_file_charpos
        let cm = init_code_map();

        let cp1 = cm.bytepos_to_file_charpos(BytePos(22));
        assert_eq!(cp1, CharPos(22));

        let cp2 = cm.bytepos_to_file_charpos(BytePos(23));
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

        let loc2 = cm.lookup_char_pos(BytePos(23));
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
        fm2.next_line(BytePos(39));
        fm2.next_line(BytePos(57));

        fm1.record_multibyte_char(BytePos(3), 3);
        fm1.record_multibyte_char(BytePos(9), 3);
        fm1.record_multibyte_char(BytePos(12), 3);
        fm1.record_multibyte_char(BytePos(15), 3);
        fm1.record_multibyte_char(BytePos(18), 3);
        fm2.record_multibyte_char(BytePos(49), 3);
        fm2.record_multibyte_char(BytePos(52), 3);
        fm2.record_multibyte_char(BytePos(57), 3);

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

        let cp3 = cm.bytepos_to_file_charpos(BytePos(55));
        assert_eq!(cp3, CharPos(12));

        let cp4 = cm.bytepos_to_file_charpos(BytePos(60));
        assert_eq!(cp4, CharPos(15));
    }
}
