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

use core::prelude::*;

use core::cmp;
use core::to_bytes;
use core::uint;
use extra::serialize::{Encodable, Decodable, Encoder, Decoder};

pub trait Pos {
    fn from_uint(n: uint) -> Self;
    fn to_uint(&self) -> uint;
}

/// A byte offset
#[deriving(Eq)]
pub struct BytePos(uint);
/// A character offset. Because of multibyte utf8 characters, a byte offset
/// is not equivalent to a character offset. The CodeMap will convert BytePos
/// values to CharPos values as necessary.
#[deriving(Eq)]
pub struct CharPos(uint);

// XXX: Lots of boilerplate in these impls, but so far my attempts to fix
// have been unsuccessful

impl Pos for BytePos {
    fn from_uint(n: uint) -> BytePos { BytePos(n) }
    fn to_uint(&self) -> uint { **self }
}

impl cmp::Ord for BytePos {
    fn lt(&self, other: &BytePos) -> bool { **self < **other }
    fn le(&self, other: &BytePos) -> bool { **self <= **other }
    fn ge(&self, other: &BytePos) -> bool { **self >= **other }
    fn gt(&self, other: &BytePos) -> bool { **self > **other }
}

impl Add<BytePos, BytePos> for BytePos {
    fn add(&self, rhs: &BytePos) -> BytePos {
        BytePos(**self + **rhs)
    }
}

impl Sub<BytePos, BytePos> for BytePos {
    fn sub(&self, rhs: &BytePos) -> BytePos {
        BytePos(**self - **rhs)
    }
}

impl to_bytes::IterBytes for BytePos {
    fn iter_bytes(&self, lsb0: bool, f: to_bytes::Cb) -> bool {
        (**self).iter_bytes(lsb0, f)
    }
}

impl Pos for CharPos {
    fn from_uint(n: uint) -> CharPos { CharPos(n) }
    fn to_uint(&self) -> uint { **self }
}

impl cmp::Ord for CharPos {
    fn lt(&self, other: &CharPos) -> bool { **self < **other }
    fn le(&self, other: &CharPos) -> bool { **self <= **other }
    fn ge(&self, other: &CharPos) -> bool { **self >= **other }
    fn gt(&self, other: &CharPos) -> bool { **self > **other }
}

impl to_bytes::IterBytes for CharPos {
    fn iter_bytes(&self, lsb0: bool, f: to_bytes::Cb) -> bool {
        (**self).iter_bytes(lsb0, f)
    }
}

impl Add<CharPos,CharPos> for CharPos {
    fn add(&self, rhs: &CharPos) -> CharPos {
        CharPos(**self + **rhs)
    }
}

impl Sub<CharPos,CharPos> for CharPos {
    fn sub(&self, rhs: &CharPos) -> CharPos {
        CharPos(**self - **rhs)
    }
}

/**
Spans represent a region of code, used for error reporting. Positions in spans
are *absolute* positions from the beginning of the codemap, not positions
relative to FileMaps. Methods on the CodeMap can be used to relate spans back
to the original source.
*/
pub struct span {
    lo: BytePos,
    hi: BytePos,
    expn_info: Option<@ExpnInfo>
}

#[deriving(Eq, Encodable, Decodable)]
pub struct spanned<T> { node: T, span: span }

impl cmp::Eq for span {
    fn eq(&self, other: &span) -> bool {
        return (*self).lo == (*other).lo && (*self).hi == (*other).hi;
    }
    fn ne(&self, other: &span) -> bool { !(*self).eq(other) }
}

impl<S:Encoder> Encodable<S> for span {
    /* Note #1972 -- spans are encoded but not decoded */
    fn encode(&self, s: &mut S) {
        s.emit_nil()
    }
}

impl<D:Decoder> Decodable<D> for span {
    fn decode(_d: &mut D) -> span {
        dummy_sp()
    }
}

impl to_bytes::IterBytes for span {
    fn iter_bytes(&self, lsb0: bool, f: to_bytes::Cb) -> bool {
        self.lo.iter_bytes(lsb0, f) &&
        self.hi.iter_bytes(lsb0, f) &&
        self.expn_info.iter_bytes(lsb0, f)
    }
}

pub fn spanned<T>(lo: BytePos, hi: BytePos, t: T) -> spanned<T> {
    respan(mk_sp(lo, hi), t)
}

pub fn respan<T>(sp: span, t: T) -> spanned<T> {
    spanned {node: t, span: sp}
}

pub fn dummy_spanned<T>(t: T) -> spanned<T> {
    respan(dummy_sp(), t)
}

/* assuming that we're not in macro expansion */
pub fn mk_sp(lo: BytePos, hi: BytePos) -> span {
    span {lo: lo, hi: hi, expn_info: None}
}

// make this a const, once the compiler supports it
pub fn dummy_sp() -> span { return mk_sp(BytePos(0), BytePos(0)); }



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
pub struct NameAndSpan {name: @str, span: Option<span>}

impl to_bytes::IterBytes for NameAndSpan {
    fn iter_bytes(&self, lsb0: bool, f: to_bytes::Cb) -> bool {
        self.name.iter_bytes(lsb0, f) && self.span.iter_bytes(lsb0, f)
    }
}

pub struct CallInfo {
    call_site: span,
    callee: NameAndSpan
}

impl to_bytes::IterBytes for CallInfo {
    fn iter_bytes(&self, lsb0: bool, f: to_bytes::Cb) -> bool {
        self.call_site.iter_bytes(lsb0, f) && self.callee.iter_bytes(lsb0, f)
    }
}

/// Extra information for tracking macro expansion of spans
pub enum ExpnInfo {
    ExpandedFrom(CallInfo)
}

impl to_bytes::IterBytes for ExpnInfo {
    fn iter_bytes(&self, lsb0: bool, f: to_bytes::Cb) -> bool {
        match *self {
            ExpandedFrom(ref call_info) => {
                0u8.iter_bytes(lsb0, f) && call_info.iter_bytes(lsb0, f)
            }
        }
    }
}

pub type FileName = @str;

pub struct FileLines
{
    file: @FileMap,
    lines: ~[uint]
}

// represents the origin of a file:
pub enum FileSubstr {
    // indicates that this is a normal standalone file:
    pub FssNone,
    // indicates that this "file" is actually a substring
    // of another file that appears earlier in the codemap
    pub FssInternal(span),
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
    /// Extra information used by qquote
    substr: FileSubstr,
    /// The complete source code
    src: @str,
    /// The start position of this source in the CodeMap
    start_pos: BytePos,
    /// Locations of lines beginnings in the source code
    lines: @mut ~[BytePos],
    /// Locations of multi-byte characters in the source code
    multibyte_chars: @mut ~[MultiByteChar],
}

impl FileMap {
    // EFFECT: register a start-of-line offset in the
    // table of line-beginnings.
    // UNCHECKED INVARIANT: these offsets must be added in the right
    // order and must be in the right places; there is shared knowledge
    // about what ends a line between this file and parse.rs
    pub fn next_line(&self, pos: BytePos) {
        // the new charpos must be > the last one (or it's the first one).
        let lines = &mut *self.lines;
        assert!((lines.len() == 0) || (lines[lines.len() - 1] < pos))
        lines.push(pos);
    }

    // get a line from the list of pre-computed line-beginnings
    pub fn get_line(&self, line: int) -> ~str {
        let begin: BytePos = self.lines[line] - self.start_pos;
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
        self.multibyte_chars.push(mbc);
    }
}

pub struct CodeMap {
    files: @mut ~[@FileMap]
}

impl CodeMap {
    pub fn new() -> CodeMap {
        CodeMap {
            files: @mut ~[],
        }
    }

    /// Add a new FileMap to the CodeMap and return it
    pub fn new_filemap(&self, filename: FileName, src: @str) -> @FileMap {
        return self.new_filemap_w_substr(filename, FssNone, src);
    }

    pub fn new_filemap_w_substr(&self,
                                filename: FileName,
                                substr: FileSubstr,
                                src: @str)
                                -> @FileMap {
        let files = &mut *self.files;
        let start_pos = if files.len() == 0 {
            0
        } else {
            let last_start = files.last().start_pos.to_uint();
            let last_len = files.last().src.len();
            last_start + last_len
        };

        let filemap = @FileMap {
            name: filename, substr: substr, src: src,
            start_pos: BytePos(start_pos),
            lines: @mut ~[],
            multibyte_chars: @mut ~[],
        };

        files.push(filemap);

        return filemap;
    }

    pub fn mk_substr_filename(&self, sp: span) -> ~str {
        let pos = self.lookup_char_pos(sp.lo);
        return fmt!("<%s:%u:%u>", pos.file.name,
                    pos.line, pos.col.to_uint());
    }

    /// Lookup source information about a BytePos
    pub fn lookup_char_pos(&self, pos: BytePos) -> Loc {
        return self.lookup_pos(pos);
    }

    pub fn lookup_char_pos_adj(&self, pos: BytePos) -> LocWithOpt {
        let loc = self.lookup_char_pos(pos);
        match (loc.file.substr) {
            FssNone =>
            LocWithOpt {
                filename: loc.file.name,
                line: loc.line,
                col: loc.col,
                file: Some(loc.file)},
            FssInternal(sp) =>
            self.lookup_char_pos_adj(
                sp.lo + (pos - loc.file.start_pos)),
        }
    }

    pub fn adjust_span(&self, sp: span) -> span {
        let line = self.lookup_line(sp.lo);
        match (line.fm.substr) {
            FssNone => sp,
            FssInternal(s) => {
                self.adjust_span(span {
                    lo: s.lo + (sp.lo - line.fm.start_pos),
                    hi: s.lo + (sp.hi - line.fm.start_pos),
                    expn_info: sp.expn_info
                })
            }
        }
    }

    pub fn span_to_str(&self, sp: span) -> ~str {
        let files = &*self.files;
        if files.len() == 0 && sp == dummy_sp() {
            return ~"no-location";
        }

        let lo = self.lookup_char_pos_adj(sp.lo);
        let hi = self.lookup_char_pos_adj(sp.hi);
        return fmt!("%s:%u:%u: %u:%u", lo.filename,
                    lo.line, lo.col.to_uint(), hi.line, hi.col.to_uint())
    }

    pub fn span_to_filename(&self, sp: span) -> FileName {
        let lo = self.lookup_char_pos(sp.lo);
        return /* FIXME (#2543) */ copy lo.file.name;
    }

    pub fn span_to_lines(&self, sp: span) -> @FileLines {
        let lo = self.lookup_char_pos(sp.lo);
        let hi = self.lookup_char_pos(sp.hi);
        let mut lines = ~[];
        for uint::range(lo.line - 1u, hi.line as uint) |i| {
            lines.push(i);
        };
        return @FileLines {file: lo.file, lines: lines};
    }

    pub fn span_to_snippet(&self, sp: span) -> ~str {
        let begin = self.lookup_byte_offset(sp.lo);
        let end = self.lookup_byte_offset(sp.hi);
        assert_eq!(begin.fm.start_pos, end.fm.start_pos);
        return begin.fm.src.slice(
                          begin.pos.to_uint(), end.pos.to_uint()).to_owned();
    }

    pub fn get_filemap(&self, filename: &str) -> @FileMap {
        for self.files.iter().advance |fm| { if filename == fm.name { return *fm; } }
        //XXjdm the following triggers a mismatched type bug
        //      (or expected function, found _|_)
        fail!(); // ("asking for " + filename + " which we don't know about");
    }
}

impl CodeMap {
    fn lookup_filemap_idx(&self, pos: BytePos) -> uint {
        let files = &*self.files;
        let len = files.len();
        let mut a = 0u;
        let mut b = len;
        while b - a > 1u {
            let m = (a + b) / 2u;
            if self.files[m].start_pos > pos {
                b = m;
            } else {
                a = m;
            }
        }
        if (a >= len) {
            fail!("position %u does not resolve to a source location", pos.to_uint())
        }

        return a;
    }

    fn lookup_line(&self, pos: BytePos) -> FileMapAndLine
    {
        let idx = self.lookup_filemap_idx(pos);
        let f = self.files[idx];
        let mut a = 0u;
        let lines = &*f.lines;
        let mut b = lines.len();
        while b - a > 1u {
            let m = (a + b) / 2u;
            if lines[m] > pos { b = m; } else { a = m; }
        }
        return FileMapAndLine {fm: f, line: a};
    }

    fn lookup_pos(&self, pos: BytePos) -> Loc {
        let FileMapAndLine {fm: f, line: a} = self.lookup_line(pos);
        let line = a + 1u; // Line numbers start at 1
        let chpos = self.bytepos_to_local_charpos(pos);
        let linebpos = f.lines[a];
        let linechpos = self.bytepos_to_local_charpos(linebpos);
        debug!("codemap: byte pos %? is on the line at byte pos %?",
               pos, linebpos);
        debug!("codemap: char pos %? is on the line at char pos %?",
               chpos, linechpos);
        debug!("codemap: byte is on line: %?", line);
        assert!(chpos >= linechpos);
        return Loc {
            file: f,
            line: line,
            col: chpos - linechpos
        };
    }

    fn span_to_str_no_adj(&self, sp: span) -> ~str {
        let lo = self.lookup_char_pos(sp.lo);
        let hi = self.lookup_char_pos(sp.hi);
        return fmt!("%s:%u:%u: %u:%u", lo.file.name,
                    lo.line, lo.col.to_uint(), hi.line, hi.col.to_uint())
    }

    fn lookup_byte_offset(&self, bpos: BytePos)
        -> FileMapAndBytePos {
        let idx = self.lookup_filemap_idx(bpos);
        let fm = self.files[idx];
        let offset = bpos - fm.start_pos;
        return FileMapAndBytePos {fm: fm, pos: offset};
    }

    // Converts an absolute BytePos to a CharPos relative to the file it is
    // located in
    fn bytepos_to_local_charpos(&self, bpos: BytePos) -> CharPos {
        debug!("codemap: converting %? to char pos", bpos);
        let idx = self.lookup_filemap_idx(bpos);
        let map = self.files[idx];

        // The number of extra bytes due to multibyte chars in the FileMap
        let mut total_extra_bytes = 0;

        for map.multibyte_chars.iter().advance |mbc| {
            debug!("codemap: %?-byte char at %?", mbc.bytes, mbc.pos);
            if mbc.pos < bpos {
                total_extra_bytes += mbc.bytes;
                // We should never see a byte position in the middle of a
                // character
                assert!(bpos == mbc.pos
                    || bpos.to_uint() >= mbc.pos.to_uint() + mbc.bytes);
            } else {
                break;
            }
        }

        CharPos(bpos.to_uint() - total_extra_bytes)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn t1 () {
        let cm = CodeMap::new();
        let fm = cm.new_filemap(@"blork.rs",@"first line.\nsecond line");
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
        let fm = cm.new_filemap(@"blork.rs",@"first line.\nsecond line");
        // TESTING *REALLY* BROKEN BEHAVIOR:
        fm.next_line(BytePos(0));
        fm.next_line(BytePos(10));
        fm.next_line(BytePos(2));
    }
}
