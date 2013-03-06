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
use core::dvec::DVec;
use core::str;
use core::to_bytes;
use core::uint;
use std::serialize::{Encodable, Decodable, Encoder, Decoder};

pub trait Pos {
    static pure fn from_uint(n: uint) -> Self;
    pure fn to_uint(&self) -> uint;
}

/// A byte offset
pub enum BytePos = uint;
/// A character offset. Because of multibyte utf8 characters, a byte offset
/// is not equivalent to a character offset. The CodeMap will convert BytePos
/// values to CharPos values as necessary.
pub enum CharPos = uint;

// XXX: Lots of boilerplate in these impls, but so far my attempts to fix
// have been unsuccessful

impl Pos for BytePos {
    static pure fn from_uint(n: uint) -> BytePos { BytePos(n) }
    pure fn to_uint(&self) -> uint { **self }
}

impl cmp::Eq for BytePos {
    pure fn eq(&self, other: &BytePos) -> bool { **self == **other }
    pure fn ne(&self, other: &BytePos) -> bool { !(*self).eq(other) }
}

impl cmp::Ord for BytePos {
    pure fn lt(&self, other: &BytePos) -> bool { **self < **other }
    pure fn le(&self, other: &BytePos) -> bool { **self <= **other }
    pure fn ge(&self, other: &BytePos) -> bool { **self >= **other }
    pure fn gt(&self, other: &BytePos) -> bool { **self > **other }
}

impl Add<BytePos, BytePos> for BytePos {
    pure fn add(&self, rhs: &BytePos) -> BytePos {
        BytePos(**self + **rhs)
    }
}

impl Sub<BytePos, BytePos> for BytePos {
    pure fn sub(&self, rhs: &BytePos) -> BytePos {
        BytePos(**self - **rhs)
    }
}

impl to_bytes::IterBytes for BytePos {
    pure fn iter_bytes(&self, +lsb0: bool, &&f: to_bytes::Cb) {
        (**self).iter_bytes(lsb0, f)
    }
}

impl Pos for CharPos {
    static pure fn from_uint(n: uint) -> CharPos { CharPos(n) }
    pure fn to_uint(&self) -> uint { **self }
}

impl cmp::Eq for CharPos {
    pure fn eq(&self, other: &CharPos) -> bool { **self == **other }
    pure fn ne(&self, other: &CharPos) -> bool { !(*self).eq(other) }
}

impl cmp::Ord for CharPos {
    pure fn lt(&self, other: &CharPos) -> bool { **self < **other }
    pure fn le(&self, other: &CharPos) -> bool { **self <= **other }
    pure fn ge(&self, other: &CharPos) -> bool { **self >= **other }
    pure fn gt(&self, other: &CharPos) -> bool { **self > **other }
}

impl to_bytes::IterBytes for CharPos {
    pure fn iter_bytes(&self, +lsb0: bool, &&f: to_bytes::Cb) {
        (**self).iter_bytes(lsb0, f)
    }
}

impl Add<CharPos,CharPos> for CharPos {
    pure fn add(&self, rhs: &CharPos) -> CharPos {
        CharPos(**self + **rhs)
    }
}

impl Sub<CharPos,CharPos> for CharPos {
    pure fn sub(&self, rhs: &CharPos) -> CharPos {
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

#[auto_encode]
#[auto_decode]
#[deriving_eq]
pub struct spanned<T> { node: T, span: span }

impl cmp::Eq for span {
    pure fn eq(&self, other: &span) -> bool {
        return (*self).lo == (*other).lo && (*self).hi == (*other).hi;
    }
    pure fn ne(&self, other: &span) -> bool { !(*self).eq(other) }
}

impl<S:Encoder> Encodable<S> for span {
    /* Note #1972 -- spans are encoded but not decoded */
    fn encode(&self, _s: &S) { }
}

impl<D:Decoder> Decodable<D> for span {
    static fn decode(_d: &D) -> span {
        dummy_sp()
    }
}

pub pure fn spanned<T>(+lo: BytePos, +hi: BytePos, +t: T) -> spanned<T> {
    respan(mk_sp(lo, hi), t)
}

pub pure fn respan<T>(sp: span, +t: T) -> spanned<T> {
    spanned {node: t, span: sp}
}

pub pure fn dummy_spanned<T>(+t: T) -> spanned<T> {
    respan(dummy_sp(), t)
}

/* assuming that we're not in macro expansion */
pub pure fn mk_sp(+lo: BytePos, +hi: BytePos) -> span {
    span {lo: lo, hi: hi, expn_info: None}
}

// make this a const, once the compiler supports it
pub pure fn dummy_sp() -> span { return mk_sp(BytePos(0), BytePos(0)); }



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
    filename: ~str,
    line: uint,
    col: CharPos,
    file: Option<@FileMap>,
}

// used to be structural records. Better names, anyone?
pub struct FileMapAndLine {fm: @FileMap, line: uint}
pub struct FileMapAndBytePos {fm: @FileMap, pos: BytePos}
pub struct NameAndSpan {name: ~str, span: Option<span>}

pub struct CallInfo {
    call_site: span,
    callee: NameAndSpan
}

/// Extra information for tracking macro expansion of spans
pub enum ExpnInfo {
    ExpandedFrom(CallInfo)
}

pub type FileName = ~str;

pub struct FileLines {
    file: @FileMap,
    lines: ~[uint]
}

pub enum FileSubstr {
    pub FssNone,
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
    src: @~str,
    /// The start position of this source in the CodeMap
    start_pos: BytePos,
    /// Locations of lines beginnings in the source code
    lines: @mut ~[BytePos],
    /// Locations of multi-byte characters in the source code
    multibyte_chars: DVec<MultiByteChar>
}

pub impl FileMap {
    // EFFECT: register a start-of-line offset in the
    // table of line-beginnings.
    // UNCHECKED INVARIANT: these offsets must be added in the right
    // order and must be in the right places; there is shared knowledge
    // about what ends a line between this file and parse.rs
    fn next_line(&self, +pos: BytePos) {
        // the new charpos must be > the last one (or it's the first one).
        assert ((self.lines.len() == 0)
                || (self.lines[self.lines.len() - 1] < pos));
        self.lines.push(pos);
    }

    // get a line from the list of pre-computed line-beginnings
    pub fn get_line(&self, line: int) -> ~str {
        unsafe {
            let begin: BytePos = self.lines[line] - self.start_pos;
            let begin = begin.to_uint();
            let end = match str::find_char_from(*self.src, '\n', begin) {
                Some(e) => e,
                None => str::len(*self.src)
            };
            str::slice(*self.src, begin, end)
        }
    }

    pub fn record_multibyte_char(&self, pos: BytePos, bytes: uint) {
        assert bytes >=2 && bytes <= 4;
        let mbc = MultiByteChar {
            pos: pos,
            bytes: bytes,
        };
        self.multibyte_chars.push(mbc);
    }
}

pub struct CodeMap {
    files: DVec<@FileMap>
}

pub impl CodeMap {
    static pub fn new() -> CodeMap {
        CodeMap {
            files: DVec()
        }
    }

    /// Add a new FileMap to the CodeMap and return it
    fn new_filemap(&self, +filename: FileName, src: @~str) -> @FileMap {
        return self.new_filemap_w_substr(filename, FssNone, src);
    }

    fn new_filemap_w_substr(
        &self,
        +filename: FileName,
        +substr: FileSubstr,
        src: @~str
    ) -> @FileMap {
        let start_pos = if self.files.len() == 0 {
            0
        } else {
            let last_start = self.files.last().start_pos.to_uint();
            let last_len = self.files.last().src.len();
            last_start + last_len
        };

        let filemap = @FileMap {
            name: filename, substr: substr, src: src,
            start_pos: BytePos(start_pos),
            lines: @mut ~[],
            multibyte_chars: DVec()
        };

        self.files.push(filemap);

        return filemap;
    }

    pub fn mk_substr_filename(&self, sp: span) -> ~str {
        let pos = self.lookup_char_pos(sp.lo);
        return fmt!("<%s:%u:%u>", pos.file.name,
                    pos.line, pos.col.to_uint());
    }

    /// Lookup source information about a BytePos
    pub fn lookup_char_pos(&self, +pos: BytePos) -> Loc {
        return self.lookup_pos(pos);
    }

    pub fn lookup_char_pos_adj(&self, +pos: BytePos) -> LocWithOpt
    {
        let loc = self.lookup_char_pos(pos);
        match (loc.file.substr) {
            FssNone =>
            LocWithOpt {
                filename: /* FIXME (#2543) */ copy loc.file.name,
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
        if self.files.len() == 0 && sp == dummy_sp() {
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
        assert begin.fm.start_pos == end.fm.start_pos;
        return str::slice(*begin.fm.src,
                          begin.pos.to_uint(), end.pos.to_uint());
    }

    pub fn get_filemap(&self, filename: ~str) -> @FileMap {
        for self.files.each |fm| { if fm.name == filename { return *fm; } }
        //XXjdm the following triggers a mismatched type bug
        //      (or expected function, found _|_)
        fail!(); // ("asking for " + filename + " which we don't know about");
    }

}

priv impl CodeMap {

    fn lookup_filemap_idx(&self, +pos: BytePos) -> uint {
        let len = self.files.len();
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
            fail!(fmt!("position %u does not resolve to a source location",
                      pos.to_uint()))
        }

        return a;
    }

    fn lookup_line(&self, pos: BytePos) -> FileMapAndLine
    {
        let idx = self.lookup_filemap_idx(pos);
        let f = self.files[idx];
        let mut a = 0u;
        let mut b = f.lines.len();
        while b - a > 1u {
            let m = (a + b) / 2u;
            if f.lines[m] > pos { b = m; } else { a = m; }
        }
        return FileMapAndLine {fm: f, line: a};
    }

    fn lookup_pos(&self, +pos: BytePos) -> Loc {
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
        assert chpos >= linechpos;
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

    fn lookup_byte_offset(&self, +bpos: BytePos)
        -> FileMapAndBytePos {
        let idx = self.lookup_filemap_idx(bpos);
        let fm = self.files[idx];
        let offset = bpos - fm.start_pos;
        return FileMapAndBytePos {fm: fm, pos: offset};
    }

    // Converts an absolute BytePos to a CharPos relative to the file it is
    // located in
    fn bytepos_to_local_charpos(&self, +bpos: BytePos) -> CharPos {
        debug!("codemap: converting %? to char pos", bpos);
        let idx = self.lookup_filemap_idx(bpos);
        let map = self.files[idx];

        // The number of extra bytes due to multibyte chars in the FileMap
        let mut total_extra_bytes = 0;

        for map.multibyte_chars.each |mbc| {
            debug!("codemap: %?-byte char at %?", mbc.bytes, mbc.pos);
            if mbc.pos < bpos {
                total_extra_bytes += mbc.bytes;
                // We should never see a byte position in the middle of a
                // character
                assert bpos == mbc.pos
                    || bpos.to_uint() >= mbc.pos.to_uint() + mbc.bytes;
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
    use util::testing::check_equal;

    #[test]
    fn t1 () {
        let cm = CodeMap::new();
        let fm = cm.new_filemap(~"blork.rs",@~"first line.\nsecond line");
        fm.next_line(BytePos(0));
        check_equal(&fm.get_line(0),&~"first line.");
        // TESTING BROKEN BEHAVIOR:
        fm.next_line(BytePos(10));
        check_equal(&fm.get_line(1),&~".");
    }

    #[test]
    #[should_fail]
    fn t2 () {
        let cm = CodeMap::new();
        let fm = cm.new_filemap(~"blork.rs",@~"first line.\nsecond line");
        // TESTING *REALLY* BROKEN BEHAVIOR:
        fm.next_line(BytePos(0));
        fm.next_line(BytePos(10));
        fm.next_line(BytePos(2));
    }
}



//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
//
