/*! A codemap is a thing that maps uints to file/line/column positions
 * in a crate. This to make it possible to represent the positions
 * with single-word things, rather than passing records all over the
 * compiler.
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

pub enum BytePos = uint;
pub enum CharPos = uint;

impl BytePos: Pos {
    static pure fn from_uint(n: uint) -> BytePos { BytePos(n) }
    pure fn to_uint(&self) -> uint { **self }
}

impl BytePos: cmp::Eq {
    pure fn eq(other: &BytePos) -> bool {
        *self == **other
    }
    pure fn ne(other: &BytePos) -> bool { !self.eq(other) }
}

impl BytePos: cmp::Ord {
    pure fn lt(other: &BytePos) -> bool { *self < **other }
    pure fn le(other: &BytePos) -> bool { *self <= **other }
    pure fn ge(other: &BytePos) -> bool { *self >= **other }
    pure fn gt(other: &BytePos) -> bool { *self > **other }
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
    pure fn iter_bytes(+lsb0: bool, f: to_bytes::Cb) {
        (*self).iter_bytes(lsb0, f)
    }
}

impl CharPos: Pos {
    static pure fn from_uint(n: uint) -> CharPos { CharPos(n) }
    pure fn to_uint(&self) -> uint { **self }
}

impl CharPos: cmp::Eq {
    pure fn eq(other: &CharPos) -> bool {
        *self == **other
    }
    pure fn ne(other: &CharPos) -> bool { !self.eq(other) }
}

impl CharPos: cmp::Ord {
    pure fn lt(other: &CharPos) -> bool { *self < **other }
    pure fn le(other: &CharPos) -> bool { *self <= **other }
    pure fn ge(other: &CharPos) -> bool { *self >= **other }
    pure fn gt(other: &CharPos) -> bool { *self > **other }
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
    pure fn iter_bytes(+lsb0: bool, f: to_bytes::Cb) {
        (*self).iter_bytes(lsb0, f)
    }
}

pub struct span {
    lo: CharPos,
    hi: CharPos,
    expn_info: Option<@ExpnInfo>
}

impl span : cmp::Eq {
    pure fn eq(other: &span) -> bool {
        return self.lo == (*other).lo && self.hi == (*other).hi;
    }
    pure fn ne(other: &span) -> bool { !self.eq(other) }
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

pub struct Loc<A: Pos> {
    file: @FileMap, line: uint, col: A
}

pub struct FilePos {
    ch: CharPos, byte: BytePos
}

impl FilePos : cmp::Eq {
    pure fn eq(other: &FilePos) -> bool {
        self.ch == (*other).ch && self.byte == (*other).byte
    }
    pure fn ne(other: &FilePos) -> bool { !self.eq(other) }
}

pub enum ExpnInfo {
    ExpandedFrom({call_site: span,
                  callie: {name: ~str, span: Option<span>}})
}

pub type FileName = ~str;

pub type LookupFn = pure fn(FilePos) -> uint;

pub struct FileLines {
    file: @FileMap,
    lines: ~[uint]
}

pub enum FileSubstr {
    pub FssNone,
    pub FssInternal(span),
    pub FssExternal({filename: ~str, line: uint, col: CharPos})
}

pub struct FileMap {
    name: FileName,
    substr: FileSubstr,
    src: @~str,
    start_pos: FilePos,
    mut lines: ~[FilePos]
}

pub impl FileMap {
    static fn new_w_substr(+filename: FileName, +substr: FileSubstr,
                           src: @~str,
                           +start_pos: FilePos)
        -> FileMap {
        return FileMap {
            name: filename, substr: substr, src: src,
            start_pos: start_pos,
            mut lines: ~[]
        };
    }

    static fn new(+filename: FileName, src: @~str,
                  +start_pos: FilePos)
        -> FileMap {
        return FileMap::new_w_substr(filename, FssNone, src,
                                     start_pos);
    }

    fn next_line(@self, +chpos: CharPos, +byte_pos: BytePos) {
        self.lines.push(FilePos {
            ch: chpos,
            byte: byte_pos + self.start_pos.byte
        });
    }

    pub fn get_line(@self, line: int) -> ~str unsafe {
        let begin: BytePos = self.lines[line].byte - self.start_pos.byte;
        let begin = begin.to_uint();
        let end = match str::find_char_from(*self.src, '\n', begin) {
            Some(e) => e,
            None => str::len(*self.src)
        };
        str::slice(*self.src, begin, end)
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

    pub fn mk_substr_filename(@self, sp: span) -> ~str {
        let pos = self.lookup_char_pos(sp.lo);
        return fmt!("<%s:%u:%u>", pos.file.name,
                    pos.line, pos.col.to_uint());
    }

    pub fn lookup_char_pos(@self, +pos: CharPos) -> Loc<CharPos> {
        pure fn lookup(pos: FilePos) -> uint { return pos.ch.to_uint(); }
        return self.lookup_pos(pos, lookup);
    }

    pub fn lookup_byte_pos(@self, +pos: BytePos) -> Loc<BytePos> {
        pure fn lookup(pos: FilePos) -> uint { return pos.byte.to_uint(); }
        return self.lookup_pos(pos, lookup);
    }

    pub fn lookup_char_pos_adj(@self, +pos: CharPos)
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
                    sp.lo + (pos - loc.file.start_pos.ch))
            }
            FssExternal(eloc) => {
                {filename: /* FIXME (#2543) */ copy eloc.filename,
                 line: eloc.line + loc.line - 1u,
                 col: if loc.line == 1u {eloc.col + loc.col} else {loc.col},
                 file: None}
            }
        }
    }

    pub fn adjust_span(@self, sp: span) -> span {
        pure fn lookup(pos: FilePos) -> uint { return pos.ch.to_uint(); }
        let line = self.lookup_line(sp.lo, lookup);
        match (line.fm.substr) {
            FssNone => sp,
            FssInternal(s) => {
                self.adjust_span(span {
                    lo: s.lo + (sp.lo - line.fm.start_pos.ch),
                    hi: s.lo + (sp.hi - line.fm.start_pos.ch),
                    expn_info: sp.expn_info
                })
            }
            FssExternal(_) => sp
        }
    }

    pub fn span_to_str(@self, sp: span) -> ~str {
        let lo = self.lookup_char_pos_adj(sp.lo);
        let hi = self.lookup_char_pos_adj(sp.hi);
        return fmt!("%s:%u:%u: %u:%u", lo.filename,
                    lo.line, lo.col.to_uint(), hi.line, hi.col.to_uint())
    }

    pub fn span_to_filename(@self, sp: span) -> FileName {
        let lo = self.lookup_char_pos(sp.lo);
        return /* FIXME (#2543) */ copy lo.file.name;
    }

    pub fn span_to_lines(@self, sp: span) -> @FileLines {
        let lo = self.lookup_char_pos(sp.lo);
        let hi = self.lookup_char_pos(sp.hi);
        let mut lines = ~[];
        for uint::range(lo.line - 1u, hi.line as uint) |i| {
            lines.push(i);
        };
        return @FileLines {file: lo.file, lines: lines};
    }

    fn lookup_byte_offset(@self, +chpos: CharPos)
        -> {fm: @FileMap, pos: BytePos} {
        pure fn lookup(pos: FilePos) -> uint { return pos.ch.to_uint(); }
        let {fm, line} = self.lookup_line(chpos, lookup);
        let line_offset = fm.lines[line].byte - fm.start_pos.byte;
        let col = chpos - fm.lines[line].ch;
        let col_offset = str::count_bytes(*fm.src,
                                          line_offset.to_uint(),
                                          col.to_uint());
        {fm: fm, pos: line_offset + BytePos(col_offset)}
    }

    pub fn span_to_snippet(@self, sp: span) -> ~str {
        let begin = self.lookup_byte_offset(sp.lo);
        let end = self.lookup_byte_offset(sp.hi);
        assert begin.fm.start_pos == end.fm.start_pos;
        return str::slice(*begin.fm.src,
                          begin.pos.to_uint(), end.pos.to_uint());
    }

    pub fn get_filemap(@self, filename: ~str) -> @FileMap {
        for self.files.each |fm| { if fm.name == filename { return *fm; } }
        //XXjdm the following triggers a mismatched type bug
        //      (or expected function, found _|_)
        fail; // ("asking for " + filename + " which we don't know about");
    }

}

priv impl CodeMap {
    fn lookup_line<A: Pos>(@self, pos: A, lookup: LookupFn)
        -> {fm: @FileMap, line: uint}
    {
        let len = self.files.len();
        let mut a = 0u;
        let mut b = len;
        while b - a > 1u {
            let m = (a + b) / 2u;
            if lookup(self.files[m].start_pos) > pos.to_uint() {
                b = m;
            } else {
                a = m;
            }
        }
        if (a >= len) {
            fail fmt!("position %u does not resolve to a source location",
                      pos.to_uint())
        }
        let f = self.files[a];
        a = 0u;
        b = vec::len(f.lines);
        while b - a > 1u {
            let m = (a + b) / 2u;
            if lookup(f.lines[m]) > pos.to_uint() { b = m; } else { a = m; }
        }
        return {fm: f, line: a};
    }

    fn lookup_pos<A: Pos Num>(@self, pos: A, lookup: LookupFn) -> Loc<A> {
        let {fm: f, line: a} = self.lookup_line(pos, lookup);
        return Loc {
            file: f,
            line: a + 1u,
            col: pos - from_uint(lookup(f.lines[a]))
        };
    }

    fn span_to_str_no_adj(@self, sp: span) -> ~str {
        let lo = self.lookup_char_pos(sp.lo);
        let hi = self.lookup_char_pos(sp.hi);
        return fmt!("%s:%u:%u: %u:%u", lo.file.name,
                    lo.line, lo.col.to_uint(), hi.line, hi.col.to_uint())
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
