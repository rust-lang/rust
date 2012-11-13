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

pub type byte_pos = uint;
pub type char_pos = uint;

pub struct span {
    lo: char_pos,
    hi: char_pos,
    expn_info: Option<@expn_info>
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

pub struct file_pos {
    ch: char_pos, byte: byte_pos
}

pub struct loc {
    file: @filemap, line: uint, col: uint
}

impl file_pos : cmp::Eq {
    pure fn eq(other: &file_pos) -> bool {
        self.ch == (*other).ch && self.byte == (*other).byte
    }
    pure fn ne(other: &file_pos) -> bool { !self.eq(other) }
}

pub enum expn_info {
    expanded_from({call_site: span,
                   callie: {name: ~str, span: Option<span>}})
}

pub type filename = ~str;

pub type lookup_fn = pure fn(file_pos) -> uint;

pub struct file_lines {
    file: @filemap,
    lines: ~[uint]
}

pub enum file_substr {
    pub fss_none,
    pub fss_internal(span),
    pub fss_external({filename: ~str, line: uint, col: uint})
}

pub struct filemap {
    name: filename,
    substr: file_substr,
    src: @~str,
    start_pos: file_pos,
    mut lines: ~[file_pos]
}

pub impl filemap {
    static fn new_w_substr(+filename: filename, +substr: file_substr,
                           src: @~str,
                           start_pos_ch: uint, start_pos_byte: uint)
        -> filemap {
        return filemap {
            name: filename, substr: substr, src: src,
            start_pos: file_pos {ch: start_pos_ch, byte: start_pos_byte},
            mut lines: ~[file_pos {ch: start_pos_ch, byte: start_pos_byte}]
        };
    }

    static fn new(+filename: filename, src: @~str,
                  start_pos_ch: char_pos, start_pos_byte: byte_pos)
        -> filemap {
        return filemap::new_w_substr(filename, fss_none, src,
                                     start_pos_ch, start_pos_byte);
    }

    fn next_line(@self, chpos: char_pos, byte_pos: byte_pos) {
        self.lines.push(file_pos {ch: chpos, byte: byte_pos + self.start_pos.byte});
    }

    pub fn get_line(@self, line: int) -> ~str unsafe {
        let begin: uint = self.lines[line].byte - self.start_pos.byte;
        let end = match str::find_char_from(*self.src, '\n', begin) {
            Some(e) => e,
            None => str::len(*self.src)
        };
        str::slice(*self.src, begin, end)
    }

}

pub struct CodeMap {
    files: DVec<@filemap>
}

pub impl CodeMap {
    static pub fn new() -> CodeMap {
        CodeMap {
            files: DVec()
        }
    }

    pub fn mk_substr_filename(@self, sp: span) -> ~str {
        let pos = self.lookup_char_pos(sp.lo);
        return fmt!("<%s:%u:%u>", pos.file.name, pos.line, pos.col);
    }

    pub fn lookup_char_pos(@self, pos: char_pos) -> loc {
        pure fn lookup(pos: file_pos) -> uint { return pos.ch; }
        return self.lookup_pos(pos, lookup);
    }

    pub fn lookup_byte_pos(@self, pos: byte_pos) -> loc {
        pure fn lookup(pos: file_pos) -> uint { return pos.byte; }
        return self.lookup_pos(pos, lookup);
    }

    pub fn lookup_char_pos_adj(@self, pos: char_pos)
        -> {filename: ~str, line: uint, col: uint, file: Option<@filemap>}
    {
        let loc = self.lookup_char_pos(pos);
        match (loc.file.substr) {
            fss_none => {
                {filename: /* FIXME (#2543) */ copy loc.file.name,
                 line: loc.line,
                 col: loc.col,
                 file: Some(loc.file)}
            }
            fss_internal(sp) => {
                self.lookup_char_pos_adj(sp.lo + (pos - loc.file.start_pos.ch))
            }
            fss_external(eloc) => {
                {filename: /* FIXME (#2543) */ copy eloc.filename,
                 line: eloc.line + loc.line - 1u,
                 col: if loc.line == 1u {eloc.col + loc.col} else {loc.col},
                 file: None}
            }
        }
    }

    pub fn adjust_span(@self, sp: span) -> span {
        pure fn lookup(pos: file_pos) -> uint { return pos.ch; }
        let line = self.lookup_line(sp.lo, lookup);
        match (line.fm.substr) {
            fss_none => sp,
            fss_internal(s) => {
                self.adjust_span(span {lo: s.lo + (sp.lo - line.fm.start_pos.ch),
                                       hi: s.lo + (sp.hi - line.fm.start_pos.ch),
                                       expn_info: sp.expn_info})}
            fss_external(_) => sp
        }
    }

    pub fn span_to_str(@self, sp: span) -> ~str {
        let lo = self.lookup_char_pos_adj(sp.lo);
        let hi = self.lookup_char_pos_adj(sp.hi);
        return fmt!("%s:%u:%u: %u:%u", lo.filename,
                    lo.line, lo.col, hi.line, hi.col)
    }

    pub fn span_to_filename(@self, sp: span) -> filename {
        let lo = self.lookup_char_pos(sp.lo);
        return /* FIXME (#2543) */ copy lo.file.name;
    }

    pub fn span_to_lines(@self, sp: span) -> @file_lines {
        let lo = self.lookup_char_pos(sp.lo);
        let hi = self.lookup_char_pos(sp.hi);
        let mut lines = ~[];
        for uint::range(lo.line - 1u, hi.line as uint) |i| {
            lines.push(i);
        };
        return @file_lines {file: lo.file, lines: lines};
    }

    fn lookup_byte_offset(@self, chpos: char_pos)
        -> {fm: @filemap, pos: byte_pos} {
        pure fn lookup(pos: file_pos) -> uint { return pos.ch; }
        let {fm, line} = self.lookup_line(chpos, lookup);
        let line_offset = fm.lines[line].byte - fm.start_pos.byte;
        let col = chpos - fm.lines[line].ch;
        let col_offset = str::count_bytes(*fm.src, line_offset, col);
        {fm: fm, pos: line_offset + col_offset}
    }

    pub fn span_to_snippet(@self, sp: span) -> ~str {
        let begin = self.lookup_byte_offset(sp.lo);
        let end = self.lookup_byte_offset(sp.hi);
        assert begin.fm.start_pos == end.fm.start_pos;
        return str::slice(*begin.fm.src, begin.pos, end.pos);
    }

    pub fn get_snippet(@self, fidx: uint, lo: uint, hi: uint) -> ~str
    {
        let fm = self.files[fidx];
        return str::slice(*fm.src, lo, hi)
    }

    pub fn get_filemap(@self, filename: ~str) -> @filemap {
        for self.files.each |fm| { if fm.name == filename { return *fm; } }
        //XXjdm the following triggers a mismatched type bug
        //      (or expected function, found _|_)
        fail; // ("asking for " + filename + " which we don't know about");
    }

}

priv impl CodeMap {
    fn lookup_line(@self, pos: uint, lookup: lookup_fn)
        -> {fm: @filemap, line: uint}
    {
        let len = self.files.len();
        let mut a = 0u;
        let mut b = len;
        while b - a > 1u {
            let m = (a + b) / 2u;
            if lookup(self.files[m].start_pos) > pos { b = m; } else { a = m; }
        }
        if (a >= len) {
            fail fmt!("position %u does not resolve to a source location", pos)
        }
        let f = self.files[a];
        a = 0u;
        b = vec::len(f.lines);
        while b - a > 1u {
            let m = (a + b) / 2u;
            if lookup(f.lines[m]) > pos { b = m; } else { a = m; }
        }
        return {fm: f, line: a};
    }

    fn lookup_pos(@self, pos: uint, lookup: lookup_fn) -> loc {
        let {fm: f, line: a} = self.lookup_line(pos, lookup);
        return loc {file: f, line: a + 1u, col: pos - lookup(f.lines[a])};
    }

    fn span_to_str_no_adj(@self, sp: span) -> ~str {
        let lo = self.lookup_char_pos(sp.lo);
        let hi = self.lookup_char_pos(sp.hi);
        return fmt!("%s:%u:%u: %u:%u", lo.file.name,
                    lo.line, lo.col, hi.line, hi.col)
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
