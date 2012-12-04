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

use dvec::DVec;
use std::serialization::{Serializable,
                         Deserializable,
                         Serializer,
                         Deserializer};

trait Pos {
    static pure fn from_uint(n: uint) -> self;
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

impl BytePos: Pos {
    static pure fn from_uint(n: uint) -> BytePos { BytePos(n) }
    pure fn to_uint(&self) -> uint { **self }
}

impl BytePos: cmp::Eq {
    pure fn eq(&self, other: &BytePos) -> bool { **self == **other }
    pure fn ne(&self, other: &BytePos) -> bool { !(*self).eq(other) }
}

impl BytePos: cmp::Ord {
    pure fn lt(&self, other: &BytePos) -> bool { **self < **other }
    pure fn le(&self, other: &BytePos) -> bool { **self <= **other }
    pure fn ge(&self, other: &BytePos) -> bool { **self >= **other }
    pure fn gt(&self, other: &BytePos) -> bool { **self > **other }
}

impl BytePos: Num {
    pure fn add(other: &BytePos) -> BytePos {
        BytePos(*self + **other)
    }
    pure fn sub(other: &BytePos) -> BytePos {
        BytePos(*self - **other)
    }
    pure fn mul(other: &BytePos) -> BytePos {
        BytePos(*self * (**other))
    }
    pure fn div(other: &BytePos) -> BytePos {
        BytePos(*self / **other)
    }
    pure fn modulo(other: &BytePos) -> BytePos {
        BytePos(*self % **other)
    }
    pure fn neg() -> BytePos {
        BytePos(-*self)
    }
    pure fn to_int() -> int { *self as int }
    static pure fn from_int(+n: int) -> BytePos { BytePos(n as uint) }
}

impl BytePos: to_bytes::IterBytes {
    pure fn iter_bytes(&self, +lsb0: bool, f: to_bytes::Cb) {
        (**self).iter_bytes(lsb0, f)
    }
}

impl CharPos: Pos {
    static pure fn from_uint(n: uint) -> CharPos { CharPos(n) }
    pure fn to_uint(&self) -> uint { **self }
}

impl CharPos: cmp::Eq {
    pure fn eq(&self, other: &CharPos) -> bool { **self == **other }
    pure fn ne(&self, other: &CharPos) -> bool { !(*self).eq(other) }
}

impl CharPos: cmp::Ord {
    pure fn lt(&self, other: &CharPos) -> bool { **self < **other }
    pure fn le(&self, other: &CharPos) -> bool { **self <= **other }
    pure fn ge(&self, other: &CharPos) -> bool { **self >= **other }
    pure fn gt(&self, other: &CharPos) -> bool { **self > **other }
}

impl CharPos: Num {
    pure fn add(other: &CharPos) -> CharPos {
        CharPos(*self + **other)
    }
    pure fn sub(other: &CharPos) -> CharPos {
        CharPos(*self - **other)
    }
    pure fn mul(other: &CharPos) -> CharPos {
        CharPos(*self * (**other))
    }
    pure fn div(other: &CharPos) -> CharPos {
        CharPos(*self / **other)
    }
    pure fn modulo(other: &CharPos) -> CharPos {
        CharPos(*self % **other)
    }
    pure fn neg() -> CharPos {
        CharPos(-*self)
    }
    pure fn to_int() -> int { *self as int }
    static pure fn from_int(+n: int) -> CharPos { CharPos(n as uint) }
}

impl CharPos: to_bytes::IterBytes {
    pure fn iter_bytes(&self, +lsb0: bool, f: to_bytes::Cb) {
        (**self).iter_bytes(lsb0, f)
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

impl span : cmp::Eq {
    pure fn eq(&self, other: &span) -> bool {
        return (*self).lo == (*other).lo && (*self).hi == (*other).hi;
    }
    pure fn ne(&self, other: &span) -> bool { !(*self).eq(other) }
}

impl<S: Serializer> span: Serializable<S> {
    /* Note #1972 -- spans are serialized but not deserialized */
    fn serialize(&self, _s: &S) { }
}

impl<D: Deserializer> span: Deserializable<D> {
    static fn deserialize(_d: &D) -> span {
        ast_util::dummy_sp()
    }
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

/// Extra information for tracking macro expansion of spans
pub enum ExpnInfo {
    ExpandedFrom({call_site: span,
                  callie: {name: ~str, span: Option<span>}})
}

pub type FileName = ~str;

pub struct FileLines {
    file: @FileMap,
    lines: ~[uint]
}

pub enum FileSubstr {
    pub FssNone,
    pub FssInternal(span),
    pub FssExternal({filename: ~str, line: uint, col: CharPos})
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
    mut lines: ~[BytePos],
    /// Locations of multi-byte characters in the source code
    multibyte_chars: DVec<MultiByteChar>
}

pub impl FileMap {
    fn next_line(&self, +pos: BytePos) {
        self.lines.push(pos);
    }

    pub fn get_line(&self, line: int) -> ~str unsafe {
        let begin: BytePos = self.lines[line] - self.start_pos;
        let begin = begin.to_uint();
        let end = match str::find_char_from(*self.src, '\n', begin) {
            Some(e) => e,
            None => str::len(*self.src)
        };
        str::slice(*self.src, begin, end)
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
    fn new_filemap(+filename: FileName, src: @~str) -> @FileMap {
        return self.new_filemap_w_substr(filename, FssNone, src);
    }

    fn new_filemap_w_substr(+filename: FileName, +substr: FileSubstr,
                            src: @~str) -> @FileMap {
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
            mut lines: ~[],
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

    pub fn lookup_char_pos_adj(&self, +pos: BytePos)
        -> {filename: ~str, line: uint, col: CharPos, file: Option<@FileMap>}
    {
        let loc = self.lookup_char_pos(pos);
        match (loc.file.substr) {
            FssNone => {
                {filename: /* FIXME (#2543) */ copy loc.file.name,
                 line: loc.line,
                 col: loc.col,
                 file: Some(loc.file)}
            }
            FssInternal(sp) => {
                self.lookup_char_pos_adj(
                    sp.lo + (pos - loc.file.start_pos))
            }
            FssExternal(eloc) => {
                {filename: /* FIXME (#2543) */ copy eloc.filename,
                 line: eloc.line + loc.line - 1u,
                 col: if loc.line == 1u {eloc.col + loc.col} else {loc.col},
                 file: None}
            }
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
            FssExternal(_) => sp
        }
    }

    pub fn span_to_str(&self, sp: span) -> ~str {
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
        fail; // ("asking for " + filename + " which we don't know about");
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
            fail fmt!("position %u does not resolve to a source location",
                      pos.to_uint())
        }

        return a;
    }

    fn lookup_line(&self, +pos: BytePos)
        -> {fm: @FileMap, line: uint}
    {
        let idx = self.lookup_filemap_idx(pos);
        let f = self.files[idx];
        let mut a = 0u;
        let mut b = vec::len(f.lines);
        while b - a > 1u {
            let m = (a + b) / 2u;
            if f.lines[m] > pos { b = m; } else { a = m; }
        }
        return {fm: f, line: a};
    }

    fn lookup_pos(&self, +pos: BytePos) -> Loc {
        let {fm: f, line: a} = self.lookup_line(pos);
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
        -> {fm: @FileMap, pos: BytePos} {
        let idx = self.lookup_filemap_idx(bpos);
        let fm = self.files[idx];
        let offset = bpos - fm.start_pos;
        return {fm: fm, pos: offset};
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

//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
//
